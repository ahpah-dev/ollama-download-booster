"""
Ollama Registry Client
~~~~~~~~~~~~~~~~~~~~~~
Client for interacting with Ollama OCI Model Registry (registry.ollama.ai).
Fetches manifests, inspects layers, and resolves direct CDN/R2 storage URLs.
"""

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

    def fetch_manifest(self, model_ref: str) -> Tuple[Dict[str, Any], Tuple[str, str, str, str]]:
        """
        Fetches the JSON manifest for a given model.
        Returns (manifest_dict, parsed_tuple).
        """
        reg, ns, model, tag = self.parse_model_ref(model_ref)
        url = f"https://{reg}/v2/{ns}/{model}/manifests/{tag}"

        last_error = None
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    return response.json(), (reg, ns, model, tag)
                elif response.status_code == 404:
                    raise ValueError(f"Model '{model_ref}' not found on {reg} (HTTP 404). Check the model name and tag.")
                else:
                    response.raise_for_status()
            except (requests.RequestException, ValueError) as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Failed to fetch manifest for '{model_ref}': {last_error}")

    def resolve_blob_download_url(self, registry: str, namespace: str, model: str, digest: str) -> str:
        """
        Resolves the direct storage URL for a blob (following 307 redirect to Cloudflare R2).
        By obtaining the direct signed CDN URL, worker threads bypass registry API gateway bottlenecks.
        """
        url = f"https://registry.ollama.ai/v2/{namespace}/{model}/blobs/{digest}"
        try:
            # We don't follow redirects immediately so we capture the signed direct R2 URL
            resp = self.session.get(url, allow_redirects=False, timeout=12)
            if resp.status_code in (301, 302, 307, 308):
                location = resp.headers.get("Location")
                if location:
                    return location
            elif resp.status_code == 200:
                return url
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
