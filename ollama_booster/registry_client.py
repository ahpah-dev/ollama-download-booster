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
        Supports:
          'llama3.2' -> ('registry.ollama.ai', 'library', 'llama3.2', 'latest')
          'llama3.2:1b' -> ('registry.ollama.ai', 'library', 'llama3.2', '1b')
          'hf.co/user/repo:Q4_K_M' -> ('hf.co', 'user', 'repo', 'Q4_K_M')
          'https://huggingface.co/user/repo' -> ('hf.co', 'user', 'repo', 'latest')
          'https://huggingface.co/user/repo/blob/main/model-Q4_K_M.gguf' -> ('hf.co', 'user', 'repo', 'Q4_K_M')
        """
        import re

        ref = model_ref.strip()

        # Clean protocol prefixes
        if ref.startswith("https://"):
            ref = ref[8:]
        elif ref.startswith("http://"):
            ref = ref[7:]

        # Normalize huggingface.co to hf.co
        if ref.startswith("huggingface.co/"):
            ref = "hf.co/" + ref[15:]

        # Handle Hugging Face direct browser URLs (tree or blob)
        if ref.startswith("hf.co/"):
            parts = ref.split("/")
            if len(parts) >= 4 and parts[3] in ("tree", "blob"):
                ns = parts[1]
                mdl = parts[2]
                tag = "latest"
                if len(parts) >= 6 and parts[3] == "blob" and parts[5].endswith(".gguf"):
                    fname = parts[5][:-5]
                    m = re.search(r"[-_.]([qQ][0-9]_[A-Za-z0-9_]+|[fF]16|[fF]32|[bB][fF]16)$", fname)
                    if m:
                        tag = m.group(1).upper()
                return "hf.co", ns, mdl, tag

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
    def is_valid_hf_token(token: Optional[str]) -> bool:
        """Validate if a string conforms to Hugging Face Access Token syntax."""
        import re
        if not token or not isinstance(token, str):
            return False
        cleaned = token.strip()
        return bool(re.match(r"^hf_[A-Za-z0-9_]{20,}$", cleaned))

    @staticmethod
    def extract_hf_token(text: Optional[str]) -> Optional[str]:
        """Extract a valid Hugging Face token from any arbitrary text or clipboard content."""
        import re
        if not text or not isinstance(text, str):
            return None
        match = re.search(r"\b(hf_[A-Za-z0-9_]{20,})\b", text.strip())
        if match:
            candidate = match.group(1).strip()
            if RegistryClient.is_valid_hf_token(candidate):
                return candidate
        return None

    @staticmethod
    def get_hf_token() -> Optional[str]:
        """
        Auto-discovers Hugging Face token across multiple system locations:
        1. Process Environment: HF_TOKEN, HUGGING_FACE_HUB_TOKEN
        2. Hugging Face CLI / Cache: ~/.cache/huggingface/token, ~/.huggingface/token
        3. Windows Registry: HKCU\\Environment\\HF_TOKEN
        4. Booster Configuration: ~/.ollama/booster_config.json
        5. huggingface_hub library get_token() (if installed)
        """
        # 1. Environment variables
        for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            val = os.getenv(env_var)
            if val and RegistryClient.is_valid_hf_token(val):
                return val.strip()

        # 2. Hugging Face standard cache paths
        for path in ("~/.cache/huggingface/token", "~/.huggingface/token"):
            p = os.path.expanduser(path)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        val = f.read().strip()
                        if RegistryClient.is_valid_hf_token(val):
                            return val
                except Exception:
                    pass

        # 3. Windows Registry (HKCU\Environment\HF_TOKEN)
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_READ) as key:
                    val, _ = winreg.QueryValueEx(key, "HF_TOKEN")
                    if val and RegistryClient.is_valid_hf_token(val):
                        return val.strip()
            except Exception:
                pass

        # 4. Ollama Booster config file
        config_path = os.path.expanduser("~/.ollama/booster_config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    val = cfg.get("hf_token", "")
                    if val and RegistryClient.is_valid_hf_token(val):
                        return val.strip()
            except Exception:
                pass

        # 5. huggingface_hub library
        try:
            from huggingface_hub import get_token as hf_get_token
            tok = hf_get_token()
            if tok and RegistryClient.is_valid_hf_token(tok):
                return tok.strip()
        except Exception:
            pass

        return None

    @staticmethod
    def save_hf_token(token: str) -> None:
        """
        Persists Hugging Face token simultaneously to:
        1. Process Environment (HF_TOKEN)
        2. Booster Config (~/.ollama/booster_config.json)
        3. Hugging Face Cache (~/.cache/huggingface/token)
        4. Windows User Registry (HKCU\\Environment\\HF_TOKEN)
        """
        cleaned = token.strip() if token else ""

        # 1. Environment
        if cleaned:
            os.environ["HF_TOKEN"] = cleaned
        else:
            os.environ.pop("HF_TOKEN", None)

        # 2. Booster Config
        config_dir = os.path.expanduser("~/.ollama")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "booster_config.json")
        cfg = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        if cleaned:
            cfg["hf_token"] = cleaned
        else:
            cfg.pop("hf_token", None)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

        # 3. Standard Hugging Face Cache Directory
        try:
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(hf_cache_dir, exist_ok=True)
            hf_token_file = os.path.join(hf_cache_dir, "token")
            if cleaned:
                with open(hf_token_file, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            elif os.path.exists(hf_token_file):
                try:
                    os.remove(hf_token_file)
                except Exception:
                    pass
        except Exception:
            pass

        # 4. Windows Registry Persistence
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE) as key:
                    if cleaned:
                        winreg.SetValueEx(key, "HF_TOKEN", 0, winreg.REG_SZ, cleaned)
                    else:
                        try:
                            winreg.DeleteValue(key, "HF_TOKEN")
                        except Exception:
                            pass
            except Exception:
                pass

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
                            f"Could not access Hugging Face model '{model_ref}' (HTTP {response.status_code}).\n\n"
                            f"• Public Hugging Face models require ZERO tokens! Check for typos in the user/repo name or quantization tag.\n"
                            f"• Hugging Face returns 401 when a repo does not exist to avoid leaking private repo names.\n"
                            f"• If this model is gated (such as Meta-Llama or Gemma), accept the license agreement at:\n"
                            f"  https://huggingface.co/{ns}/{model}\n"
                            f"  and connect your free Hugging Face token in Settings."
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


def is_valid_hf_token(token: Optional[str]) -> bool:
    """Helper module-level function to check if a token has valid HF syntax."""
    return RegistryClient.is_valid_hf_token(token)


def extract_hf_token(text: Optional[str]) -> Optional[str]:
    """Helper module-level function to extract HF token from text."""
    return RegistryClient.extract_hf_token(text)


def get_hf_token() -> Optional[str]:
    """Helper module-level function to retrieve Hugging Face token."""
    return RegistryClient.get_hf_token()


def save_hf_token(token: str) -> None:
    """Helper module-level function to save Hugging Face token."""
    RegistryClient.save_hf_token(token)
