"""
Ollama Download Booster
~~~~~~~~~~~~~~~~~~~~~~~
A high-performance multi-stream download accelerator for Ollama models.
Boosts download throughput by up to 10x using concurrent HTTP range connections,
direct-seek disk streaming, and instant local Ollama registration.
"""

__version__ = "1.0.0"
__author__ = "ahpah-dev"

from .models_manager import OllamaManager
from .registry_client import RegistryClient
from .booster_engine import BoosterEngine
from .proxy_server import BoosterProxyServer

__all__ = [
    "OllamaManager",
    "RegistryClient",
    "BoosterEngine",
    "BoosterProxyServer",
]
