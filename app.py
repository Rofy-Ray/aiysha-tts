import os
import logging
import uuid
import torch
import soundfile as sf
from flask import Flask, request, jsonify
from google.cloud import storage
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

try:
    spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
    vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
except Exception as e:
    logger.error(f"Error loading TTS models: {e}")
    raise

app = Flask(__name__)

def generate_tts(text):
    """
    Generate audio from text using NeMo FastPitch and HifiGan models.
    
    Args:
        text (str): Input text to convert to speech
    
    Returns:
        str: Path to generated WAV file
    """
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")

        parsed = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
        
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        
        audio_numpy = audio.to('cpu').detach().numpy()[0]
        
        output_filename = f"{uuid.uuid4()}.wav"
        
        sf.write(output_filename, audio_numpy, 22050)
        
        return output_filename

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise

def save_audio_to_gcs(audio_file_path, text):
    """
    Save audio file to Google Cloud Storage.
    
    Args:
        audio_file_path (str): Path to audio file
        text (str): Original text (for context)
    
    Returns:
        str: Public URL of uploaded file
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        blob = bucket.blob(os.path.basename(audio_file_path))
        blob.upload_from_filename(audio_file_path)
        
        os.unlink(audio_file_path)
        
        return blob.public_url
    
    except Exception as e:
        logger.error(f"GCS upload error: {e}")
        raise

@app.route('/tts', methods=['POST'])
def tts_handler():
    """
    Flask endpoint for text-to-speech conversion.
    
    Expects JSON with 'text' key.
    Returns audio URL or error.
    """
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json or 'text' not in request_json:
            return jsonify({'error': 'No text provided'}), 400
        
        text = request_json['text']
        
        audio_file = generate_tts(text)
        
        audio_url = save_audio_to_gcs(audio_file, text)
        
        return jsonify({
            'audio_url': audio_url,
            'text': text
        })
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))