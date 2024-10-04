# Vocal chat
This is a basic webapp allowing two way vocal interactions with a small local large language model. The code was written by OpenAI O1 in iterative steps.
Large language models are known to have issues such as hallucinations. 
Accuracy and Reliability: While the model strives to provide accurate and helpful information, it may generate incorrect or misleading responses. Users should verify information from reliable sources before acting upon it.
No Professional Advice: This webapp does not offer professional advice (medical, legal, financial, etc.). For such matters, please consult a qualified professional.
Privacy and Data Security: Be mindful of the information you share during interactions. Although the model operates locally, ensure that your device is secure to protect your privacy.
Content Responsibility: Users are responsible for the content they input and the use of the generated responses. The developers are not liable for any misuse or consequences arising from the use of this webapp.
Updates and Maintenance: The webapp and the underlying model may receive updates to improve performance and security. However, no guarantees are made regarding uninterrupted access or error-free operation.
Limitation of Liability: In no event shall the developers or affiliates be liable for any direct, indirect, incidental, special, or consequential damages arising out of the use or inability to use this webapp.
By using this webapp, you acknowledge that you have read, understood, and agree to abide by this disclaimer. If you do not agree with any part of this disclaimer, please refrain from using the webapp. 

# Setup Guide

Follow the steps below to install and set up **Ollama** with **LLaMA 3.2** and **Whisper.cpp** on macOS.

---

## Install Ollama

Download and install **Ollama** from the official website:

- [Download Ollama for Mac](https://ollama.com/download/mac)

---

## Pull LLaMA 3.2 with 1 Billion Parameters

Open your terminal and execute the following command:

```bash
ollama pull llama3.2:1b
```

---

## Check It Is Functional

Run the model to ensure it's working correctly:

```bash
ollama run llama3.2:1b
```

After running the above command, type any query. The model should respond accordingly.

---

## Download Whisper.cpp

Clone the **Whisper.cpp** repository:

```bash
git clone https://github.com/ggerganov/whisper.cpp
```

---

## Compile Whisper.cpp

Navigate to the cloned directory and compile the project:

```bash
cd whisper.cpp
make
```

---

## Download the Model

Download the `ggml-base.en.bin` model and place it in the `whisper.cpp/models` directory:

```bash
curl -L -o ./models/ggml-base.en.bin "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin?download=true"
```

---

## Move Whisper.cpp to the Home Folder

Move the **Whisper.cpp** directory to your home folder. Replace `<your_user_name>` with your actual macOS username:

```bash
mv whisper.cpp /Users/<your_user_name>/whisper.cpp
```

---

## Install Python Packages

Install the required Python packages using `pip`. It's recommended to use a virtual environment.

1. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install the packages:**

    ```bash
    pip install flask flask-session openai
    ```

---

## Run the script

Execute your Flask script. After running, take note of the address provided in the terminal output.

```bash
python3 Ollama_webapp.py
```

**Example Output:**

```
 * Serving Flask app 'Ollama_webapp'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
```

**Access the Application:**

- Open your web browser.
- Navigate to `http://127.0.0.1:5000` or the address provided in your terminal.

---

# License

[MIT](LICENSE)

