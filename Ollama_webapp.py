from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
from openai import OpenAI
import ssl
import wave
import sys

# SSL context adjustments (if needed)
ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

# Define folder to store uploaded audio files

home_dir = os.path.expanduser("~")
os.path.join(home_dir,"Downloads","uploads")

UPLOAD_FOLDER = os.path.join(home_dir,"Downloads","uploads")  # Update to your desired path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Limit the maximum allowed payload to 16 megabytes.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Desired sample rate
DESIRED_SAMPLE_RATE = 16000

def get_sample_rate(wav_path):
    try:
        with wave.open(wav_path, 'rb') as wf:
            return wf.getframerate()
    except wave.Error as e:
        print(f"Error: Failed to read WAV file '{wav_path}': {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: WAV file not found at '{wav_path}'", file=sys.stderr)
        return None

def convert_to_wav(input_path, output_path):
    """
    Converts an audio file to WAV format with PCM 16-bit little-endian encoding,
    16kHz sample rate, and mono channel using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-i", input_path,
        "-ar", str(DESIRED_SAMPLE_RATE),  # Set sample rate to 16kHz
        "-ac", "1",                       # Set number of audio channels to 1 (mono)
        "-c:a", "pcm_s16le",              # Set audio codec to PCM 16-bit little-endian
        output_path
    ]
    print("Converting audio to WAV format with 16kHz sample rate and PCM 16-bit encoding using ffmpeg...")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Conversion successful. Converted file saved as '{output_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: FFmpeg conversion failed: {e.stderr}", file=sys.stderr)
        return False

def verify_wav_format(wav_path):
    """
    Verifies that the WAV file starts with the RIFF header.
    """
    try:
        with open(wav_path, 'rb') as f:
            header = f.read(4)
            if header != b'RIFF':
                print(f"Error: WAV file '{wav_path}' does not start with RIFF header.", file=sys.stderr)
                return False
        return True
    except Exception as e:
        print(f"Error: Could not verify WAV file '{wav_path}': {e}", file=sys.stderr)
        return False

def transcribe_with_whisper_cpp(filepath):
    home_dir = os.path.expanduser('~')
    whisper_dir = os.path.join(home_dir, 'whisper.cpp')
    whisper_cpp_executable = os.path.join(whisper_dir, 'main')
    model_path = os.path.join(whisper_dir, 'models', 'ggml-base.en.bin')

    # Check if necessary directories and files exist
    if not os.path.isdir(whisper_dir):
        print(f"Error: whisper.cpp directory not found at {whisper_dir}", file=sys.stderr)
        return None
    if not os.path.isfile(whisper_cpp_executable):
        print(f"Error: Whisper executable not found at {whisper_cpp_executable}", file=sys.stderr)
        return None
    if not os.access(whisper_cpp_executable, os.X_OK):
        print(f"Error: Whisper executable is not executable. Please set execute permissions using `chmod +x {whisper_cpp_executable}`", file=sys.stderr)
        return None
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        return None

    # Check the sample rate of the audio file
    current_sample_rate = get_sample_rate(filepath)
    if current_sample_rate is None:
        return None
    print(f"Current sample rate: {current_sample_rate} Hz")

    # Ensure sample rate is 16kHz
    if current_sample_rate != DESIRED_SAMPLE_RATE:
        print("Error: Sample rate is not 16kHz after conversion. This should not happen.", file=sys.stderr)
        return None
    else:
        print("Sample rate is confirmed to be 16kHz.")

    # Define the transcription output file
    transcription_file = filepath + '.txt'

    # Command to execute whisper.cpp
    cmd = [
        whisper_cpp_executable,
        "-m", model_path,
        "-f", filepath,
        "-otxt",
        "--language", "en",  # assuming we want English
        "--threads", "4"     # number of threads to use, optional
    ]

    print("\nExecuting command:", " ".join(cmd))
    print("Working directory:", whisper_dir)
    print("Using model:", model_path)
    print("Input audio file:", filepath)

    try:
        result = subprocess.run(
            cmd,
            cwd=whisper_dir,
            capture_output=True,
            text=True,
            timeout=120  # Increased timeout to 120 seconds
        )

        if result.returncode != 0:
            print("Whisper.cpp encountered an error:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        if not os.path.isfile(transcription_file):
            print("Error: Transcription file not found.", file=sys.stderr)
            print("Whisper.cpp output:", result.stdout, file=sys.stderr)
            return None

        with open(transcription_file, 'r') as f:
            transcription = f.read().strip()

        print("Transcription successful.")
        return transcription

    except subprocess.TimeoutExpired:
        print("Error: The transcription process timed out.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during transcription: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up transcription file if you don't need to keep it
        if os.path.isfile(transcription_file):
            os.remove(transcription_file)
            print(f"Cleaned up transcription file '{transcription_file}'")

def get_completion(messages):
    response = client.chat.completions.create(
        model="llama3.2:1b",
        messages=messages
    )
    return response.choices[0].message.content

def text_to_speech(text):
    # Use the macOS 'say' command to convert text to speech
    subprocess.run(['say', text])

# Use a global variable to store messages
messages = [
    {"role": "system", "content": "You are a helpful assistant. Write short responses of 3 sentences or less."}
]

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Initialize user_input
        user_input = ""

        # Check if audio file was uploaded
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                user_input = "No audio file selected."
            else:
                filename = secure_filename(audio_file.filename)
                original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(original_filepath)
                print(f"Uploaded audio file saved as '{original_filepath}'")

                # Define a unique path for the WAV converted file to avoid overwriting
                wav_filename = os.path.splitext(filename)[0] + "_converted.wav"
                wav_filepath = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)

                # Convert the uploaded audio to WAV with 16kHz sample rate and PCM 16-bit encoding
                conversion_success = convert_to_wav(original_filepath, wav_filepath)

                # Remove the original uploaded file to save space
                try:
                    os.remove(original_filepath)
                    print(f"Removed original uploaded file '{original_filepath}'")
                except Exception as e:
                    print(f"Warning: Could not remove original file '{original_filepath}': {e}", file=sys.stderr)

                if conversion_success and os.path.isfile(wav_filepath):
                    # Verify WAV format
                    if verify_wav_format(wav_filepath):
                        # Transcribe the audio using whisper.cpp
                        transcription = transcribe_with_whisper_cpp(wav_filepath)
                        if transcription:
                            user_input = transcription
                        else:
                            user_input = "Sorry, I couldn't transcribe your audio."
                    else:
                        user_input = "Sorry, the audio format is invalid."

                    # Clean up the converted WAV file
                    try:
                        os.remove(wav_filepath)
                        print(f"Removed converted WAV file '{wav_filepath}'")
                    except Exception as e:
                        print(f"Warning: Could not remove WAV file '{wav_filepath}': {e}", file=sys.stderr)
                else:
                    user_input = "Sorry, there was an error processing your audio file."
        else:
            # Handle text input
            user_input = request.form.get('message', '')

        if user_input:
            # Append user's message to messages
            messages.append({"role": "user", "content": user_input})

            # Get assistant response
            assistant_response = get_completion(messages)
            messages.append({"role": "assistant", "content": assistant_response})

            # Convert assistant's response to speech
            text_to_speech(assistant_response)

        return redirect(url_for('chat'))

    return render_template('chat.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=True)
