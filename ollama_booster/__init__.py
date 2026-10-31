"""
Ollama Download Booster
~~~~~~~~~~~~~~~~~~~~~~~
A high-performance multi-stream download accelerator and runtime inference booster for Ollama models.
Boosts download throughput by up to 10x using concurrent HTTP range connections,
and accelerates model inference using Flash Attention, memory residency, and CPU core tuning.
"""

__version__ = "1.1.0"
__author__ = "ahpah-dev"

from .models_manager import OllamaManager
from .registry_client import RegistryClient
from .booster_engine import BoosterEngine
from .proxy_server import BoosterProxyServer
from .runtime_booster import RuntimeBooster

__all__ = [
    "OllamaManager",
    "RegistryClient",
    "BoosterEngine",
    "BoosterProxyServer",
    "RuntimeBooster",
]
