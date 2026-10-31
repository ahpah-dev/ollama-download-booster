"""
Ollama Registry Client
~~~~~~~~~~~~~~~~~~~~~~
Client for interacting with Ollama OCI Model Registry (registry.ollama.ai).
Fetches manifests, inspects layers, and resolves direct CDN/R2 storage URLs.
"""

import os
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Tuple, Optional


class RegistryClient:
    """Interacts with Ollama registry endpoints to fetch manifests and blobs."""

    DEFAULT_REGISTRY = "registry.ollama.ai"
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 OllamaBooster/1.0"
    )

    def __init__(self, registry: Optional[str] = None):
        self.registry = registry or self.DEFAULT_REGISTRY
        self.session = requests.Session()

        # Resilient connection pool
        retries = Retry(
            total=4,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json, "
                "application/vnd.oci.image.manifest.v1+json, */*"
            ),
        })

    @staticmethod
    def parse_model_ref(model_ref: str) -> Tuple[str, str, str, str]:
        """
        Parses full model string into (registry, namespace, model, tag).
        Examples:
          'llama3.2' -> ('registry.ollama.ai', 'library', 'llama3.2', 'latest')
          'llama3.2:1b' -> ('registry.ollama.ai', 'library', 'llama3.2', '1b')
          'myuser/mymodel:7b' -> ('registry.ollama.ai', 'myuser', 'mymodel', '7b')
          'custom.registry/user/model:tag' -> ('custom.registry', 'user', 'model', 'tag')
        """
        ref = model_ref.strip()
        registry = RegistryClient.DEFAULT_REGISTRY

        # Check for host prefix (e.g. host:port/ or host.domain/)
        parts = ref.split("/", 1)
        if len(parts) == 2 and ("." in parts[0] or ":" in parts[0]):
            registry = parts[0]
            ref = parts[1]

        # Extract namespace and model
        if "/" in ref:
            namespace, rest = ref.split("/", 1)
        else:
            namespace = "library"
            rest = ref

        # Extract model and tag
        if ":" in rest:
            model, tag = rest.split(":", 1)
        else:
            model = rest
            tag = "latest"

        return registry, namespace, model, tag

    @staticmethod
    def get_hf_token() -> Optional[str]:
        """Find Hugging Face token from environment, cache, or booster config."""
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token.strip()
        cache_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        return t
            except Exception:
                pass
        config_path = os.path.expanduser("~/.ollama/booster_config.json")
        if os.path.isfile(config_path):
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("hf_token", "").strip() or None
            except Exception:
                pass
        return None

    @staticmethod
    def save_hf_token(token: str) -> None:
        """Saves Hugging Face token to ~/.ollama/booster_config.json and session env."""
        config_dir = os.path.expanduser("~/.ollama")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "booster_config.json")
        cfg = {}
        if os.path.isfile(config_path):
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        cfg["hf_token"] = token.strip()
        try:
            import json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass
        if token.strip():
            os.environ["HF_TOKEN"] = token.strip()


def get_hf_token() -> Optional[str]:
    return RegistryClient.get_hf_token()


def save_hf_token(token: str) -> None:
    RegistryClient.save_hf_token(token)

    def fetch_manifest(self, model_ref: str) -> Tuple[Dict[str, Any], Tuple[str, str, str, str]]:
        """
        Fetches the JSON manifest for a given model.
        Returns (manifest_dict, parsed_tuple).
        """
        reg, ns, model, tag = self.parse_model_ref(model_ref)
        url = f"https://{reg}/v2/{ns}/{model}/manifests/{tag}"

        headers = {}
        hf_token = self.get_hf_token()
        if reg == "hf.co" and hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        last_error = None
        for attempt in range(3):
            try:
                response = self.session.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    return response.json(), (reg, ns, model, tag)
                elif response.status_code in (401, 403):
                    if reg == "hf.co":
                        raise PermissionError(
                            f"Access unauthorized for Hugging Face model '{model_ref}' (HTTP {response.status_code}).\n"
                            f"• If this model is gated (like LiquidAI/LFM, Meta-Llama), accept the license agreement at:\n"
                            f"  https://huggingface.co/{ns}/{model}\n"
                            f"• Then configure your Hugging Face Access Token in the app settings or via export HF_TOKEN=hf_..."
                        )
                    else:
                        raise PermissionError(
                            f"Access unauthorized for model '{model_ref}' on {reg} (HTTP {response.status_code}).\n"
                            f"This model may be private or restricted."
                        )
                elif response.status_code == 404:
                    raise ValueError(f"Model '{model_ref}' not found on {reg} (HTTP 404). Check the model name and tag.")
                elif response.status_code == 412:
                    raise RuntimeError(
                        f"Model '{model_ref}' requires a newer version of Ollama. "
                        f"Please update Ollama at https://ollama.com/download"
                    )
                else:
                    response.raise_for_status()
            except (requests.RequestException, ValueError, PermissionError, RuntimeError) as e:
                last_error = e
                if isinstance(e, (PermissionError, ValueError, RuntimeError)):
                    raise
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Failed to fetch manifest for '{model_ref}': {last_error}")

    def resolve_blob_download_url(self, registry: str, namespace: str, model: str, digest: str) -> str:
        """
        Resolves the direct storage URL for a blob (following redirects to Cloudflare R2 / AWS CloudFront).
        Supports registry.ollama.ai, hf.co (Hugging Face), and custom OCI registries.
        """
        url = f"https://{registry}/v2/{namespace}/{model}/blobs/{digest}"
        headers = {}
        hf_token = self.get_hf_token()
        if registry == "hf.co" and hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        try:
            resp = self.session.get(url, headers=headers, allow_redirects=True, stream=True, timeout=15)
            if resp.status_code in (200, 206):
                return resp.url
        except Exception:
            pass
        return url

    @staticmethod
    def get_manifest_stats(manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize size and layers in manifest."""
        layers = manifest.get("layers", [])
        total_size = sum(layer.get("size", 0) for layer in layers)
        
        # Categorize layers
        model_layer = None
        other_layers = []
        for l in layers:
            media = l.get("mediaType", "")
            if "model" in media:
                model_layer = l
            else:
                other_layers.append(l)

        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "total_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "num_layers": len(layers),
            "model_layer": model_layer,
            "other_layers": other_layers,
            "config": manifest.get("config", {})
        }
