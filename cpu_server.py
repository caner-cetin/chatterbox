import os
import io
import time
import sys
import argparse

cpu_count = os.cpu_count() or 4
threads_for_tts = max(1, cpu_count - 2)
os.environ["OMP_NUM_THREADS"] = str(threads_for_tts)
os.environ["MKL_NUM_THREADS"] = str(threads_for_tts)

import torch
import torchaudio as ta
from flask import Flask, request, jsonify, Response
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from gunicorn.app.wsgiapp import WSGIApplication
from chatterbox.tts_turbo import ChatterboxTurboTTS

console = Console()
device = "cpu"

torch.set_num_threads(threads_for_tts)
torch.set_num_interop_threads(1)
torch.set_flush_denormal(True)

console.print(Panel(
    Text(f"CPU threads: {threads_for_tts} | Interop: 1 | OMP/MKL: {threads_for_tts}", style="bold green"),
    title="CPU Configuration",
    border_style="green"
))

REFERENCE_AUDIO_PATH = os.getenv(
    "REFERENCE_AUDIO_PATH",
    "/home/cansu/Downloads/Voices/GLaDOS/00_part1_entry-2.wav"
)

app = Flask(__name__)
app.config['model'] = None
app.config['model_initialized'] = False
app.config['device'] = device


def initialize_model():
    if app.config['model_initialized']:
        return
    
    console.print(Panel(
        Text("Initializing model...", style="bold yellow"),
        title="Model Initialization",
        border_style="yellow"
    ))
    
    t0 = time.perf_counter()
    logger.info("Loading ChatterboxTurboTTS model...")
    model = ChatterboxTurboTTS.from_pretrained(device)
    logger.success(f"Model loaded in {time.perf_counter() - t0:.2f}s")
    
    t0 = time.perf_counter()
    logger.info(f"Pre-computing conditionals from: {REFERENCE_AUDIO_PATH}")
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(
            f"Reference audio file not found: {REFERENCE_AUDIO_PATH}. "
            f"Please set REFERENCE_AUDIO_PATH environment variable."
        )
    
    model.prepare_conditionals(REFERENCE_AUDIO_PATH)
    logger.success(f"Conditionals computed in {time.perf_counter() - t0:.2f}s")
    
    t0 = time.perf_counter()
    logger.info("Compiling model with torch.compile...")
    try:
        model.t3.tfmr = torch.compile(model.t3.tfmr, mode="reduce-overhead")
        logger.success(f"Model compiled in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"torch.compile failed (requires g++): {e}")
    
    t0 = time.perf_counter()
    logger.info("Warming up model (first run triggers JIT compilation)...")
    try:
        with torch.inference_mode():
            dummy_wav = model.generate("Hello, this is a warmup.")
        logger.success(f"Warmup completed in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
    
    app.config['model'] = model
    app.config['model_initialized'] = True
    logger.success("Model initialization complete")


def cleanup_model():
    if app.config['model'] is not None:
        logger.info("Cleaning up model...")
        del app.config['model']
        app.config['model'] = None
        app.config['model_initialized'] = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Model cleanup complete")


@app.before_request
def ensure_model_initialized():
    if not app.config['model_initialized']:
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


@app.teardown_appcontext
def teardown_model(error):
    pass


@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    model = app.config.get('model')
    
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
        t0 = time.perf_counter()
        with torch.inference_mode():
            wav_tensor = model.generate(text)
        t_inference = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        buffer = io.BytesIO()
        ta.save(buffer, wav_tensor, model.sr, format="wav")
        buffer.seek(0)
        wav_bytes = buffer.read()
        t_encode = time.perf_counter() - t0
        
        logger.info(f"Generated {len(text)} chars | inference: {t_inference:.2f}s | encode: {t_encode:.2f}s")
        
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
    model = app.config.get('model')
    if model is None:
        return jsonify({"status": "not_ready", "message": "Model not initialized"}), 503
    return jsonify({"status": "ready", "device": app.config['device'], "threads": threads_for_tts})


def daemonize():
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"Fork failed: {e}")
        sys.exit(1)
    
    os.chdir("/")
    os.setsid()
    os.umask(0)
    
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"Second fork failed: {e}")
        sys.exit(1)
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    si = open(os.devnull, 'r')
    so = open(os.devnull, 'a+')
    se = open(os.devnull, 'a+')
    
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())


class StandaloneApplication(WSGIApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()
    
    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)
    
    def on_starting(self, server):
        logger.info("Gunicorn worker starting...")
        with app.app_context():
            try:
                initialize_model()
            except Exception as e:
                logger.critical(f"Failed to initialize model on worker start: {e}")
                raise
    
    def on_exit(self, server):
        logger.info("Gunicorn worker shutting down...")
        with app.app_context():
            cleanup_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatterboxTTS CPU Server')
    parser.add_argument('--port', type=int, default=int(os.getenv("PORT", 5000)),
                        help='Port to run server on (default: 5000)')
    parser.add_argument('--daemon', action='store_true',
                        help='Run server in background (detached from stdin)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes (default: 1)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Worker timeout in seconds (default: 120)')
    args = parser.parse_args()
    
    if args.daemon:
        logger.info("Daemonizing server...")
        daemonize()
        logger.info("Server running in background")
    
    console.print(Panel(
        Text(f"Starting Gunicorn server on port {args.port} with {args.workers} worker(s)...", style="bold blue"),
        title="Server Startup",
        border_style="blue"
    ))
    
    options = {
        'bind': f'0.0.0.0:{args.port}',
        'workers': args.workers,
        'timeout': args.timeout,
        'worker_class': 'sync',
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info',
    }
    
    StandaloneApplication(app, options).run()
