"""
Ollama Manager
~~~~~~~~~~~~~~
Handles local Ollama environment detection, models directory resolution,
manifest parsing, and Ollama CLI interoperability.
"""

import os
import sys
import json
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple


class OllamaManager:
    """Manages local Ollama model directories and CLI interactions."""

    def __init__(self, custom_models_dir: Optional[str] = None):
        if custom_models_dir:
            self.models_dir = os.path.abspath(custom_models_dir)
        else:
            self.models_dir = self._detect_models_dir()

        self.blobs_dir = os.path.join(self.models_dir, "blobs")
        self.manifests_dir = os.path.join(self.models_dir, "manifests")
        self._ensure_dirs()

    @staticmethod
    def _detect_models_dir() -> str:
        """Determines the default Ollama models storage path."""
        env_dir = os.getenv("OLLAMA_MODELS")
        if env_dir and os.path.exists(env_dir):
            return env_dir

        if sys.platform == "win32":
            user_profile = os.getenv("USERPROFILE", os.path.expanduser("~"))
            return os.path.join(user_profile, ".ollama", "models")
        return os.path.expanduser("~/.ollama/models")

    def _ensure_dirs(self) -> None:
        """Ensure blobs and manifests directories exist."""
        os.makedirs(self.blobs_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)

    @staticmethod
    def is_ollama_installed() -> bool:
        """Check if `ollama` CLI is available in PATH."""
        return shutil.which("ollama") is not None

    @staticmethod
    def get_ollama_version() -> str:
        """Retrieve installed Ollama version string."""
        if not OllamaManager.is_ollama_installed():
            return "Not Installed"
        try:
            res = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            return res.stdout.strip() or res.stderr.strip() or "Unknown"
        except Exception:
            return "Unknown"

    def get_manifest_path(self, registry: str, namespace: str, model: str, tag: str) -> str:
        """Return the target file path for a model's manifest."""
        return os.path.join(self.manifests_dir, registry, namespace, model, tag)

    def get_blob_path(self, digest: str) -> str:
        """Return the target file path for a content-addressed blob."""
        # Ollama replaces ':' with '-' for digest filenames (e.g. sha256-abc...)
        clean_name = digest.replace(":", "-")
        return os.path.join(self.blobs_dir, clean_name)

    def is_blob_downloaded(self, digest: str, expected_size: Optional[int] = None) -> bool:
        """Check if a blob exists locally and has the expected size."""
        path = self.get_blob_path(digest)
        if not os.path.isfile(path):
            return False
        if expected_size is not None and expected_size > 0:
            return os.path.getsize(path) == expected_size
        return True

    def list_installed_models(self) -> List[Dict[str, any]]:
        """
        List all locally recognized models by scanning the manifest tree.
        Returns a list of dicts with model info.
        """
        models = []
        if not os.path.exists(self.manifests_dir):
            return models

        for reg in os.listdir(self.manifests_dir):
            reg_dir = os.path.join(self.manifests_dir, reg)
            if not os.path.isdir(reg_dir):
                continue
            for ns in os.listdir(reg_dir):
                ns_dir = os.path.join(reg_dir, ns)
                if not os.path.isdir(ns_dir):
                    continue
                for mdl in os.listdir(ns_dir):
                    mdl_dir = os.path.join(ns_dir, mdl)
                    if not os.path.isdir(mdl_dir):
                        continue
                    for tag in os.listdir(mdl_dir):
                        tag_path = os.path.join(mdl_dir, tag)
                        if os.path.isfile(tag_path):
                            model_info = self._parse_manifest_file(reg, ns, mdl, tag, tag_path)
                            if model_info:
                                models.append(model_info)
        return models

    def _parse_manifest_file(self, reg: str, ns: str, mdl: str, tag: str, path: str) -> Optional[Dict[str, any]]:
        """Parse local manifest JSON to extract size, layers, and digest info."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            total_size = sum(layer.get("size", 0) for layer in data.get("layers", []))
            config_digest = data.get("config", {}).get("digest", "")
            short_id = config_digest.split(":")[-1][:12] if ":" in config_digest else "unknown"

            name = f"{mdl}:{tag}" if ns == "library" else f"{ns}/{mdl}:{tag}"
            mtime = os.path.getmtime(path)

            return {
                "name": name,
                "registry": reg,
                "namespace": ns,
                "model": mdl,
                "tag": tag,
                "id": short_id,
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 1),
                "size_gb": round(total_size / (1024 * 1024 * 1024), 2),
                "modified": mtime,
                "manifest_path": path,
            }
        except Exception:
            return None

    def delete_model(self, name: str) -> Tuple[bool, str]:
        """
        Delete a model. Uses `ollama rm` if available, or removes manifest directly.
        """
        if self.is_ollama_installed():
            try:
                res = subprocess.run(
                    ["ollama", "rm", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False
                )
                if res.returncode == 0:
                    return True, f"Successfully removed {name} via Ollama"
            except Exception as e:
                pass

        # Manual manifest removal fallback
        # Parse name e.g. "llama3.2:latest" or "user/model:tag"
        if "/" in name:
            ns_part, rest = name.split("/", 1)
        else:
            ns_part, rest = "library", name

        if ":" in rest:
            mdl_part, tag_part = rest.split(":", 1)
        else:
            mdl_part, tag_part = rest, "latest"

        reg = "registry.ollama.ai"
        mpath = self.get_manifest_path(reg, ns_part, mdl_part, tag_part)
        if os.path.exists(mpath):
            try:
                os.remove(mpath)
                return True, f"Removed manifest {mpath}"
            except Exception as e:
                return False, f"Failed to delete manifest: {e}"

        return False, f"Model {name} not found"

    def run_model_terminal(self, name: str) -> None:
        """Launch `ollama run <name>` in an interactive terminal."""
        if sys.platform == "win32":
            subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", f"ollama run {name}"])
        elif sys.platform == "darwin":
            apple_script = f'tell application "Terminal" to do script "ollama run {name}"'
            subprocess.Popen(["osascript", "-e", apple_script])
        else:
            # Linux
            for term in ["x-terminal-emulator", "gnome-terminal", "konsole", "xterm"]:
                if shutil.which(term):
                    if term == "gnome-terminal":
                        subprocess.Popen([term, "--", "ollama", "run", name])
                    else:
                        subprocess.Popen([term, "-e", f"ollama run {name}"])
                    return
            subprocess.Popen(["ollama", "run", name])

    def run_model_boosted(self, name: str) -> None:
        """Launch `ollama run <name>` with Flash Attention, Keep-Alive, and High Priority."""
        from .runtime_booster import RuntimeBooster
        rb = RuntimeBooster()
        rb.launch_boosted_terminal(name)
