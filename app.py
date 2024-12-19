from flask import Flask, render_template, request, jsonify
import whisper
import os
import subprocess
import warnings
import tempfile

# Suppress warnings for FP16 on unsupported hardware
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# FFmpeg executable path
ffmpeg_path = r"C:/ffmpeg/bin/ffmpeg.exe"

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model once to optimize performance
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded successfully.")

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    if not os.path.isfile(ffmpeg_path):
        raise FileNotFoundError(f"FFmpeg executable not found at {ffmpeg_path}.")
    try:
        subprocess.run([ffmpeg_path, '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("FFmpeg command failed. Check FFmpeg installation.") from e

def transcribe_audio(file_path):
    """Transcribe audio and detect language."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        result = model.transcribe(file_path, task="transcribe")
        detected_language = result.get('language', 'Unknown')
        transcription = result.get('text', 'Transcription failed')
        return transcription, detected_language
    except Exception as e:
        raise RuntimeError(f"An error occurred during transcription: {str(e)}")

@app.route('/')
def index():
    """Render the home page."""
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and transcribe audio."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name
        file.save(temp_path)

    try:
        # Check FFmpeg
        check_ffmpeg()
        
        # Transcribe the audio
        transcription, language = transcribe_audio(temp_path)

        # Return transcription results
        return jsonify({
            "transcription": transcription,
            "language": language
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Ensure FFmpeg is accessible before starting the app
    try:
        check_ffmpeg()
        app.run(debug=True)
    except Exception as e:
        print("Error starting application:", str(e))
