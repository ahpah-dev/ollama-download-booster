"""
Booster Local Accelerator Proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A lightweight local HTTP server and proxy running on localhost:11435.
Provides a REST interface for triggering accelerated model pulls and serving
locally cached / accelerated blobs.
"""

import os
import json
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any

from .booster_engine import BoosterEngine
from .models_manager import OllamaManager


class ProxyHandler(BaseHTTPRequestHandler):
    """Handles incoming booster requests."""

    engine: BoosterEngine = None
    manager: OllamaManager = None

    def log_message(self, format, *args):
        # Suppress standard logging to keep terminal clean
        pass

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status" or self.path == "/":
            self._send_json(200, {
                "name": "Ollama Download Booster Proxy",
                "status": "active",
                "current_speed_mb": round(self.engine.current_speed_mb, 2),
                "peak_speed_mb": round(self.engine.peak_speed_mb, 2),
                "total_downloaded_mb": round(self.engine.total_downloaded / (1024 * 1024), 2),
                "ollama_installed": self.manager.is_ollama_installed(),
                "models_dir": self.manager.models_dir
            })
        elif self.path == "/models":
            models = self.manager.list_installed_models()
            self._send_json(200, {"models": models, "count": len(models)})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/boost" or self.path == "/pull":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body) if body else {}
                model = payload.get("model")
                threads = int(payload.get("concurrency", 16))

                if not model:
                    self._send_json(400, {"error": "Missing 'model' field in JSON request"})
                    return

                # Launch download in background thread
                def bg_pull():
                    self.engine.download_model(model, concurrency=threads)

                t = threading.Thread(target=bg_pull, daemon=True)
                t.start()

                self._send_json(202, {
                    "status": "pulling",
                    "model": model,
                    "concurrency": threads,
                    "message": f"Boosted pull initiated for {model} with {threads} parallel streams"
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path == "/cancel":
            self.engine.cancel()
            self._send_json(200, {"status": "cancelled"})
        else:
            self._send_json(404, {"error": "Unknown POST route"})


class BoosterProxyServer:
    """Manages the background HTTP Booster service."""

    def __init__(self, port: int = 11435, manager: Optional[OllamaManager] = None):
        self.port = port
        self.manager = manager or OllamaManager()
        self.engine = BoosterEngine(ollama_manager=self.manager)
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False

    def is_port_in_use(self) -> bool:
        """Check if target port is already taken."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self) -> bool:
        """Start the proxy server in a background thread."""
        if self.is_running:
            return True

        if self.is_port_in_use():
            return False

        ProxyHandler.engine = self.engine
        ProxyHandler.manager = self.manager

        try:
            self.server = HTTPServer(("127.0.0.1", self.port), ProxyHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.is_running = True
            return True
        except Exception:
            return False

    def stop(self) -> None:
        """Shutdown proxy server."""
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
        self.is_running = False
