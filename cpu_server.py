import os
import io
import torch
import torchaudio as ta
from flask import Flask, request, jsonify, Response
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from chatterbox.tts_turbo import ChatterboxTurboTTS

console = Console()
device = "cpu"

cpu_count = os.cpu_count() or 4
threads_for_tts = max(1, cpu_count - 2)
torch.set_num_threads(threads_for_tts)
torch.set_flush_denormal(True)

console.print(Panel(
    Text(f"Using {threads_for_tts} CPU threads for TTS (reserved 2 cores for AzuraCast)", style="bold green"),
    title="CPU Configuration",
    border_style="green"
))

REFERENCE_AUDIO_PATH = os.getenv(
    "REFERENCE_AUDIO_PATH",
    "/home/cansu/Downloads/Voices/GLaDOS/00_part1_entry-2.wav"
)

app = Flask(__name__)
model = None


def initialize_model():
    global model
    
    logger.info("Loading ChatterboxTurboTTS model...")
    model = ChatterboxTurboTTS.from_pretrained(device)
    logger.success(f"Model loaded on device: {device}")
    
    logger.info(f"Pre-computing conditionals from reference audio: {REFERENCE_AUDIO_PATH}")
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(
            f"Reference audio file not found: {REFERENCE_AUDIO_PATH}. "
            f"Please set REFERENCE_AUDIO_PATH environment variable."
        )
    
    model.prepare_conditionals(REFERENCE_AUDIO_PATH)
    logger.success("Conditionals pre-computed successfully")
    
    logger.info("Compiling model with torch.compile for...")
    try:
        model.t3.tfmr = torch.compile(model.t3.tfmr, mode="reduce-overhead")
        logger.success("Model compiled successfully")
    except Exception as e:
        logger.warning(f"torch.compile failed (will use eager mode): {e}")
    
    logger.info("Warming up model with dummy inference...")
    try:
        with torch.inference_mode():
            dummy_wav = model.generate("Hello, this is a warmup.")
        logger.success("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


console.print(Panel(
    Text("Initializing model at startup...", style="bold yellow"),
    title="Model Initialization",
    border_style="yellow"
))

try:
    initialize_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    raise

ALLOWED_ORIGIN = "https://jukebox.cansu.dev"


@app.before_request
def handle_cors():
    origin = request.headers.get('Origin')
    
    if request.method == 'OPTIONS':
        if origin == ALLOWED_ORIGIN:
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
            response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Max-Age', '3600')
            return response
        elif origin:
            return jsonify({"error": "Origin not allowed"}), 403
        return Response()
    
    if origin and origin != ALLOWED_ORIGIN:
        return jsonify({"error": "Origin not allowed"}), 403


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin == ALLOWED_ORIGIN:
        response.headers.add('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = str(text).strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    
    try:
        with torch.inference_mode():
            wav_tensor = model.generate(text)
        
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        buffer = io.BytesIO()
        ta.save(buffer, wav_tensor, model.sr, format="wav")
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        return Response(
            wav_bytes,
            mimetype="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated.wav",
                "Content-Type": "audio/wav"
            }
        )
    
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        return jsonify({"error": f"Audio generation failed: {str(e)}"}), 500


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if model is None:
        return jsonify({"status": "not_ready", "message": "Model not initialized"}), 503
    return jsonify({"status": "ready", "device": device})


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    console.print(Panel(
        Text(f"Starting Flask server on port {port}...", style="bold blue"),
        title="Server Startup",
        border_style="blue"
    ))
    app.run(host="0.0.0.0", port=port, debug=False)
