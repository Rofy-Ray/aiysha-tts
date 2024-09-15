import os
import subprocess
import tempfile
import logging
import uuid
import argparse
from google.cloud import storage
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.DEBUG)

RIVA_SERVER = os.getenv('RIVA_SERVER')
RIVA_FUNCTION_ID = os.getenv('RIVA_FUNCTION_ID')
RIVA_API_KEY = os.getenv('RIVA_API_KEY')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

app = Flask(__name__)

def run_riva_tts(text, output_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=RIVA_SERVER)
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--metadata", nargs=2, action="append")
    parser.add_argument("--text")
    parser.add_argument("--voice", default='English-US.Female-1')
    parser.add_argument("--output")

    args_list = [
        "--server", RIVA_SERVER,
        "--use-ssl",
        "--metadata", "function-id", RIVA_FUNCTION_ID,
        "--metadata", "authorization", f"Bearer {RIVA_API_KEY}",
        "--text", text,
        "--voice", 'English-US.Female-1',
        "--output", output_file
    ]

    args, unknown_args = parser.parse_known_args(args_list)

    command = ["python", "talk.py"] + args_list

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.debug(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        logging.error(f"Error output: {e.output}")
        raise

def save_audio_to_gcs(audio_file_path, text):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    filename = f"{uuid.uuid4()}.wav"
    blob = bucket.blob(filename)
    blob.upload_from_filename(audio_file_path)
    
    return blob.public_url

@app.route('/tts', methods=['POST'])
def tts_handler():
    request_json = request.get_json(silent=True)
    
    if not request_json or 'text' not in request_json:
        return jsonify({'error': 'No text provided'}), 400
    
    text = request_json['text']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        output_file = temp_file.name
    
    try:
        run_riva_tts(text, output_file)
        audio_url = save_audio_to_gcs(output_file, text)
        
        return jsonify({
            'audio_url': audio_url,
            'text': text
        })
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))