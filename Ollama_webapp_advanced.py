from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, send_file
import os
import subprocess
from werkzeug.utils import secure_filename
from openai import OpenAI
import ssl
import wave
import sys
import uuid
import random
from datetime import datetime
import tempfile
import shutil
import io
import zipfile

# SSL context adjustments (if needed)
ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # Replace with your actual API key if required
)

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
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-i", input_path,
        "-ar", str(DESIRED_SAMPLE_RATE),  # Set sample rate to 16kHz
        "-ac", "1",  # Set number of audio channels to 1 (mono)
        "-c:a", "pcm_s16le",  # Set audio codec to PCM 16-bit little-endian
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

    current_sample_rate = get_sample_rate(filepath)
    if current_sample_rate is None:
        return None
    print(f"Current sample rate: {current_sample_rate} Hz")

    if current_sample_rate != DESIRED_SAMPLE_RATE:
        print("Error: Sample rate is not 16kHz after conversion. This should not happen.", file=sys.stderr)
        return None
    else:
        print("Sample rate is confirmed to be 16kHz.")

    transcription_file = filepath + '.txt'

    cmd = [
        whisper_cpp_executable,
        "-m", model_path,
        "-f", filepath,
        "-otxt",
        "--language", "en",
        "--threads", "4"
    ]

    print("\nExecuting command:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=whisper_dir,
            capture_output=True,
            text=True,
            timeout=120
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
        if os.path.isfile(transcription_file):
            os.remove(transcription_file)
            print(f"Cleaned up transcription file '{transcription_file}'")

def get_completion(messages):
    response = client.chat.completions.create(
        model="llama3.2:1b",
        messages=messages
    )
    return response.choices[0].message.content

def get_upload_folder():
    default_path = os.path.join(os.path.expanduser("~"), "Downloads", "uploads")
    save_path = session.get('save_path', '').strip()

    # Use the default path if save_path is not provided or is empty
    if not save_path:
        save_path = default_path

    os.makedirs(save_path, exist_ok=True)
    return save_path

def text_to_speech(text, filename_without_extension, play_audio=True):
    upload_folder = get_upload_folder()
    aiff_path = os.path.join(upload_folder, filename_without_extension + '.aiff')
    wav_path = os.path.join(upload_folder, filename_without_extension + '.wav')
    command = ['say', '-o', aiff_path, text]
    print(f"Converting text to speech and saving to '{aiff_path}'...")
    try:
        subprocess.run(command, check=True)
        print("Text-to-speech conversion to AIFF successful.")

        conversion_command = [
            "ffmpeg",
            "-y",
            "-i", aiff_path,
            "-ar", str(DESIRED_SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]
        print(f"Converting AIFF to WAV 16kHz and saving to '{wav_path}'...")
        subprocess.run(conversion_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Conversion to WAV successful.")

        if play_audio:
            play_command = ['afplay', wav_path]
            subprocess.run(play_command, check=True)
            print(f"Played audio '{wav_path}' on the server.")

        os.remove(aiff_path)
        print(f"Removed temporary AIFF file '{aiff_path}'")

        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"Error: Text-to-speech conversion or playback failed: {e}", file=sys.stderr)
        return None

def get_word_count(text):
    return len(text.split())

@app.before_request
def initialize_session():
    if 'messages' not in session:
        session['messages'] = [
            {"role": "system", "content": session.get('system_prompt', "You are a helpful assistant. Write short responses of 3 sentences or less.")}
        ]
    if 'conversation_id' not in session:
        session['conversation_id'] = '{:02d}'.format(random.randint(10, 99))
    if 'sequence_number' not in session:
        session['sequence_number'] = 0

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('settings'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        session['play_audio'] = request.form.get('play_audio') == 'on'
        session['story_mode'] = request.form.get('story_mode') == 'on'
        session['story_duration'] = int(request.form.get('story_duration', '0'))
        session['system_prompt'] = request.form.get('system_prompt', 'You are a helpful assistant. Write short responses of 3 sentences or less.')

        save_path = request.form.get('save_path', '').strip()
        if save_path:
            if os.path.isdir(save_path):
                session['save_path'] = save_path
            else:
                flash("The provided save path does not exist. Using default path.")
                session['save_path'] = ''
        else:
            session['save_path'] = ''

        session['messages'] = [
            {"role": "system", "content": session['system_prompt']}
        ]
        return redirect(url_for('chat'))

    if 'play_audio' not in session:
        session['play_audio'] = True
    if 'story_mode' not in session:
        session['story_mode'] = False
    if 'story_duration' not in session:
        session['story_duration'] = 0
    if 'system_prompt' not in session:
        session['system_prompt'] = "You are a helpful assistant. Write short responses of 3 sentences or less."
    if 'save_path' not in session:
        session['save_path'] = os.path.join(os.path.expanduser("~"), "Downloads", "uploads")

    return render_template('settings_advanced.html', 
                           play_audio=session['play_audio'], 
                           story_mode=session['story_mode'],
                           story_duration=session['story_duration'],
                           system_prompt=session['system_prompt'],
                           save_path=session['save_path'])

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'messages' not in session:
        session['messages'] = [
            {"role": "system", "content": session.get('system_prompt', "You are a helpful assistant. Write short responses of 3 sentences or less.")}
        ]

    messages = session['messages']

    if request.method == 'POST':
        user_input = ""

        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                user_input = "No audio file selected."
            else:
                temp_filename = secure_filename(audio_file.filename)
                temp_filepath = os.path.join(get_upload_folder(), temp_filename)
                audio_file.save(temp_filepath)
                print(f"Uploaded audio file saved as '{temp_filepath}'")

                session['sequence_number'] += 1
                sequence_number = session['sequence_number']

                now = datetime.now()
                day_month = now.strftime('%d%m')

                filename = f"{day_month}_{session['conversation_id']}_{sequence_number}_USER.wav"
                filepath = os.path.join(get_upload_folder(), filename)

                conversion_success = convert_to_wav(temp_filepath, filepath)

                try:
                    os.remove(temp_filepath)
                    print(f"Removed temporary uploaded file '{temp_filepath}'")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file '{temp_filepath}': {e}", file=sys.stderr)

                if conversion_success and os.path.isfile(filepath):
                    if verify_wav_format(filepath):
                        transcription = transcribe_with_whisper_cpp(filepath)
                        if transcription:
                            user_input = transcription
                            messages.append({"role": "user", "content": transcription, "audio_file": os.path.basename(filepath)})
                        else:
                            user_input = "Sorry, I couldn't transcribe your audio."
                    else:
                        user_input = "Sorry, the audio format is invalid."
                else:
                    user_input = "Sorry, there was an error processing your audio file."
        else:
            user_input = request.form.get('message', '').strip()
            if user_input:
                messages.append({"role": "user", "content": user_input})

        if user_input:
            play_audio = session.get('play_audio', True)
            story_mode = session.get('story_mode', False)
            story_duration_minutes = session.get('story_duration', 0)
            system_prompt = session.get('system_prompt', "You are a helpful assistant. Write short responses of 3 sentences or less.")

            if not any(msg.get('role') == 'user' and msg.get('content') == user_input for msg in messages):
                messages.append({"role": "user", "content": user_input})

            if story_mode and story_duration_minutes > 0:
                words_per_minute = 150
                target_word_count = story_duration_minutes * words_per_minute
                total_word_count = 0
                story_messages = messages.copy()

                while total_word_count < target_word_count:
                    assistant_response = get_completion(story_messages)
                    response_word_count = get_word_count(assistant_response)
                    total_word_count += response_word_count

                    message_entry = {"role": "assistant", "content": assistant_response}
                    session['sequence_number'] += 1
                    sequence_number = session['sequence_number']
                    now = datetime.now()
                    day_month = now.strftime('%d%m')
                    filename = f"{day_month}_{session['conversation_id']}_{sequence_number}_AI"

                    audio_path = text_to_speech(assistant_response, filename, play_audio=play_audio)
                    if audio_path:
                        message_entry['audio_file'] = os.path.basename(audio_path)

                    messages.append(message_entry)
                    story_messages.append(message_entry)
                    story_messages.append({"role": "user", "content": "Please continue."})

                session['messages'] = messages

            else:
                assistant_response = get_completion(messages)
                message_entry = {"role": "assistant", "content": assistant_response}
                session['sequence_number'] += 1
                sequence_number = session['sequence_number']
                now = datetime.now()
                day_month = now.strftime('%d%m')
                filename = f"{day_month}_{session['conversation_id']}_{sequence_number}_AI"

                audio_path = text_to_speech(assistant_response, filename, play_audio=play_audio)
                if audio_path:
                    message_entry['audio_file'] = os.path.basename(audio_path)

                messages.append(message_entry)
                session['messages'] = messages

        return redirect(url_for('chat'))

    return render_template('chat_advanced.html', messages=session['messages'])

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(get_upload_folder(), filename, as_attachment=True)

@app.route('/reset', methods=['POST'])
def reset_conversation():
    session['messages'] = [
        {"role": "system", "content": session.get('system_prompt', "You are a helpful assistant. Write short responses of 3 sentences or less.")}
    ]
    session['conversation_id'] = '{:02d}'.format(random.randint(10, 99))
    session['sequence_number'] = 0
    return redirect(url_for('chat'))

@app.route('/export', methods=['POST'])
def export_conversation():
    messages = session.get('messages', [])
    if not messages:
        flash('No conversation to export.')
        return redirect(url_for('chat'))

    audio_files = []
    for message in messages:
        if 'audio_file' in message:
            file_path = os.path.join(get_upload_folder(), message['audio_file'])
            audio_files.append(file_path)
    if not audio_files:
        flash('No audio files to export.')
        return redirect(url_for('chat'))

    temp_dir = tempfile.mkdtemp()

    try:
        # Create silence file
        silence_file = os.path.join(temp_dir, 'silence.wav')
        command = [
            'ffmpeg',
            '-y',
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=mono:sample_rate=16000',
            '-t', '1',
            '-c:a', 'pcm_s16le',
            silence_file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Silence file created at '{silence_file}'")

        files_to_concat = []
        for i, file_path in enumerate(audio_files):
            files_to_concat.append(f"file '{file_path}'")
            if i < len(audio_files) - 1:
                files_to_concat.append(f"file '{silence_file}'")

        concat_list_file = os.path.join(temp_dir, 'concat_list.txt')
        with open(concat_list_file, 'w') as f:
            for line in files_to_concat:
                f.write(line + '\n')

        now = datetime.now()
        day_month = now.strftime('%d%m')
        conversation_id = session.get('conversation_id', '00')
        output_filename = f"{day_month}_{conversation_id}.wav"
        output_filepath = os.path.join(temp_dir, output_filename)

        command = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_file,
            '-c', 'pcm_s16le',
            output_filepath
        ]
        print("Running ffmpeg command to concatenate audio files...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error concatenating audio files: {result.stderr}", file=sys.stderr)
            flash('Error exporting conversation.')
            return redirect(url_for('chat'))
        else:
            print(f"Conversation exported successfully to '{output_filepath}'")

        # Create text file with conversation
        text_filename = f"{day_month}_{conversation_id}.txt"
        text_filepath = os.path.join(temp_dir, text_filename)

        with open(text_filepath, 'w') as f:
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                if role == 'user':
                    f.write(f"User: {content}\n")
                elif role == 'assistant':
                    f.write(f"AI: {content}\n")

        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            zipf.write(output_filepath, arcname=output_filename)
            zipf.write(text_filepath, arcname=text_filename)

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f"{day_month}_{conversation_id}.zip")

    except Exception as e:
        print(f"Error exporting conversation: {e}", file=sys.stderr)
        flash('Error exporting conversation.')
        return redirect(url_for('chat'))
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
