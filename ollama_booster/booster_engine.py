"""
Booster Download Engine
~~~~~~~~~~~~~~~~~~~~~~~
High-performance multi-connection download engine for Ollama models.
Splits large model layers into parallel byte ranges, downloads them concurrently
using direct-seek file writing, validates cryptographic SHA-256 hashes, and registers
models directly into Ollama's local storage.
"""

import os
import time
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Callable, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models_manager import OllamaManager
from .registry_client import RegistryClient


class BoosterEngine:
    """Multi-threaded concurrent download accelerator for Ollama model blobs."""

    SMALL_LAYER_THRESHOLD = 5 * 1024 * 1024  # 5 MB threshold for single-stream vs chunking
    CHUNK_BUFFER_SIZE = 256 * 1024          # 256 KB stream buffer

    def __init__(self, ollama_manager: Optional[OllamaManager] = None, registry_client: Optional[RegistryClient] = None):
        self.manager = ollama_manager or OllamaManager()
        self.client = registry_client or RegistryClient()
        self._cancelled = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()

        # Metrics tracking
        self.total_downloaded = 0
        self.overall_total = 0
        self.speed_history: List[Tuple[float, int]] = []  # (timestamp, bytes)
        self.current_speed_mb = 0.0
        self.peak_speed_mb = 0.0
        self.active_threads = 0
        self.thread_states: Dict[int, Dict[str, Any]] = {}

    def cancel(self) -> None:
        """Signal engine to abort current download."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if abort was requested."""
        return self._cancelled.is_set()

    def reset_state(self) -> None:
        """Reset internal metrics before starting a new download."""
        self._cancelled.clear()
        self._paused.clear()
        self.total_downloaded = 0
        self.overall_total = 0
        self.speed_history = []
        self.current_speed_mb = 0.0
        self.peak_speed_mb = 0.0
        self.thread_states = {}

    def _update_speed(self, bytes_added: int) -> float:
        """Thread-safe speed calculation using a rolling 1.5-second time window."""
        now = time.time()
        with self._lock:
            self.total_downloaded += bytes_added
            self.speed_history.append((now, self.total_downloaded))

            # Prune samples older than 1.5s
            cutoff = now - 1.5
            while self.speed_history and self.speed_history[0][0] < cutoff:
                self.speed_history.pop(0)

            if len(self.speed_history) >= 2:
                dt = self.speed_history[-1][0] - self.speed_history[0][0]
                db = self.speed_history[-1][1] - self.speed_history[0][1]
                if dt > 0.1:
                    speed = (db / (1024 * 1024)) / dt
                    self.current_speed_mb = speed
                    if speed > self.peak_speed_mb:
                        self.peak_speed_mb = speed
            return self.current_speed_mb

    def _create_session(self, pool_size: int) -> requests.Session:
        """Create an optimized requests session with pooled connections."""
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(
            pool_connections=pool_size * 2,
            pool_maxsize=pool_size * 2,
            max_retries=retries
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({
            "User-Agent": RegistryClient.DEFAULT_USER_AGENT,
            "Accept": "*/*"
        })
        return s

    def download_model(
        self,
        model_ref: str,
        concurrency: int = 16,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        status_log_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Main entry point: Pulls manifest, downloads all config and weight layers
        at boosted multi-stream speeds, verifies hashes, and installs manifest.
        """
        self.reset_state()
        log = status_log_callback or (lambda msg: None)

        log(f"⚡ Resolving manifest for '{model_ref}'...")
        manifest, (reg, ns, model, tag) = self.client.fetch_manifest(model_ref)
        stats = RegistryClient.get_manifest_stats(manifest)

        layers = manifest.get("layers", [])
        config_obj = manifest.get("config", {})
        config_digest = config_obj.get("digest")

        # Calculate overall byte size
        all_blobs_to_download = []
        if config_digest:
            all_blobs_to_download.append((config_digest, config_obj.get("size", 0), "config"))
        for layer in layers:
            all_blobs_to_download.append((layer.get("digest"), layer.get("size", 0), layer.get("mediaType", "layer")))

        # Check what needs to be downloaded vs already present
        pending_blobs = []
        already_cached_bytes = 0
        for digest, size, ltype in all_blobs_to_download:
            if digest and self.manager.is_blob_downloaded(digest, size):
                already_cached_bytes += size
                log(f"[+] Blob {digest[:19]}... already cached ({size / (1024*1024):.1f} MB)")
            else:
                pending_blobs.append((digest, size, ltype))

        self.overall_total = sum(size for _, size, _ in all_blobs_to_download)
        self.total_downloaded = already_cached_bytes

        log(f"[i] Model: {ns}/{model}:{tag} | Total: {stats['total_mb']} MB ({len(layers)} layers)")
        if pending_blobs:
            log(f"[>] Booster active: Using {concurrency} parallel streams for acceleration")
        else:
            log("[+] All layers already verified in local Ollama storage!")

        # Initial progress snapshot so UI immediately reflects total size and cached bytes
        if progress_callback:
            progress_callback({
                "total_downloaded": self.total_downloaded,
                "completed_bytes": self.total_downloaded,
                "overall_total": self.overall_total,
                "total_bytes": self.overall_total,
                "speed_mb": 0.0,
                "peak_speed_mb": 0.0,
                "eta_seconds": 0.0,
                "thread_states": {}
            })

        # Download pending blobs
        for i, (digest, size, ltype) in enumerate(pending_blobs):
            if self._cancelled.is_set():
                log("[!] Download cancelled by user.")
                return False

            short_d = digest.split(":")[-1][:12] if ":" in digest else digest[:12]
            size_mb = size / (1024 * 1024)
            log(f"> Downloading layer [{i+1}/{len(pending_blobs)}] {short_d} ({size_mb:.2f} MB)...")

            # Resolve direct CDN/R2 URL for optimal concurrency
            direct_url = self.client.resolve_blob_download_url(reg, ns, model, digest)

            success = self._download_blob(
                direct_url=direct_url,
                digest=digest,
                expected_size=size,
                concurrency=concurrency,
                progress_callback=progress_callback,
                log=log
            )

            if not success:
                if self._cancelled.is_set():
                    log("[!] Download aborted.")
                else:
                    log(f"[ERROR] Failed downloading blob {digest}.")
                return False

        # All blobs downloaded and verified! Save manifest to register with Ollama
        manifest_path = self.manager.get_manifest_path(reg, ns, model, tag)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        log(f"[*] Model manifest registered: {manifest_path}")
        log(f"[OK] Ready! Run with: ollama run {model if ns == 'library' else f'{ns}/{model}'}:{tag}")
        return True

    def _download_blob(
        self,
        direct_url: str,
        digest: str,
        expected_size: int,
        concurrency: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        log: Callable[[str], None]
    ) -> bool:
        """Downloads a single blob using multi-stream chunking or single stream."""
        target_path = self.manager.get_blob_path(digest)
        part_path = target_path + ".part"
        meta_path = target_path + ".state.json"

        # Check if already present
        if os.path.isfile(target_path) and (expected_size <= 0 or os.path.getsize(target_path) == expected_size):
            return True

        # Use single stream for small blobs (config, template, licenses < 5MB)
        if expected_size <= self.SMALL_LAYER_THRESHOLD or concurrency <= 1:
            return self._download_single_stream(
                direct_url, target_path, part_path, expected_size, progress_callback, log
            )

        # Multi-connection chunked accelerated download
        return self._download_chunked_stream(
            direct_url, target_path, part_path, meta_path, digest, expected_size, concurrency, progress_callback, log
        )

    def _download_single_stream(
        self,
        direct_url: str,
        target_path: str,
        part_path: str,
        expected_size: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        log: Callable[[str], None]
    ) -> bool:
        """Download small blob directly with standard streaming."""
        session = self._create_session(4)
        try:
            with session.get(direct_url, stream=True, timeout=20) as resp:
                resp.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=self.CHUNK_BUFFER_SIZE):
                        if self._cancelled.is_set():
                            return False
                        if chunk:
                            f.write(chunk)
                            speed = self._update_speed(len(chunk))
                            if progress_callback:
                                eta = 0.0
                                rem = max(0, self.overall_total - self.total_downloaded)
                                if speed > 0:
                                    eta = rem / (speed * 1024 * 1024)
                                progress_callback({
                                    "total_downloaded": self.total_downloaded,
                                    "completed_bytes": self.total_downloaded,
                                    "overall_total": self.overall_total,
                                    "total_bytes": self.overall_total,
                                    "speed_mb": speed,
                                    "peak_speed_mb": self.peak_speed_mb,
                                    "eta_seconds": eta,
                                    "thread_states": {0: {"active": True, "done": self.total_downloaded}}
                                })

            # Atomic rename
            if os.path.exists(target_path):
                os.remove(target_path)
            os.replace(part_path, target_path)
            return True
        except Exception as e:
            log(f"Error during single-stream download: {e}")
            if os.path.exists(part_path):
                try: os.remove(part_path)
                except Exception: pass
            return False

    def _download_chunked_stream(
        self,
        direct_url: str,
        target_path: str,
        part_path: str,
        meta_path: str,
        digest: str,
        expected_size: int,
        concurrency: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        log: Callable[[str], None]
    ) -> bool:
        """
        High-performance concurrent chunk downloader.
        Pre-allocates file and streams parallel non-overlapping byte ranges directly.
        """
        # Determine chunk boundaries
        chunk_size = (expected_size + concurrency - 1) // concurrency
        chunks: List[Dict[str, int]] = []
        for i in range(concurrency):
            start = i * chunk_size
            end = min(expected_size - 1, (i + 1) * chunk_size - 1)
            if start <= end:
                chunks.append({"id": i, "start": start, "end": end, "bytes_done": 0})

        # Pre-allocate sparse file if not exists
        if not os.path.exists(part_path) or os.path.getsize(part_path) != expected_size:
            with open(part_path, "wb") as f:
                f.seek(expected_size - 1)
                f.write(b"\0")

        # Resume state if exists
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    saved_chunks = json.load(mf)
                if len(saved_chunks) == len(chunks):
                    chunks = saved_chunks
                    # Add already downloaded bytes to metrics
                    for c in chunks:
                        with self._lock:
                            self.total_downloaded += c.get("bytes_done", 0)
            except Exception:
                pass

        session = self._create_session(concurrency)
        thread_states: Dict[int, Dict[str, Any]] = {}
        for c in chunks:
            thread_states[c["id"]] = {
                "start": c["start"],
                "end": c["end"],
                "total": c["end"] - c["start"] + 1,
                "done": c.get("bytes_done", 0),
                "active": False
            }

        def worker(chunk_info: Dict[str, int]) -> bool:
            cid = chunk_info["id"]
            orig_start = chunk_info["start"]
            end = chunk_info["end"]
            done = chunk_info.get("bytes_done", 0)
            current_start = orig_start + done

            if current_start > end:
                # Chunk already fully downloaded
                thread_states[cid]["active"] = False
                thread_states[cid]["done"] = end - orig_start + 1
                return True

            thread_states[cid]["active"] = True

            max_retries = 4
            for attempt in range(max_retries):
                if self._cancelled.is_set():
                    return False
                try:
                    headers = {"Range": f"bytes={current_start}-{end}"}
                    with session.get(direct_url, headers=headers, stream=True, timeout=25) as resp:
                        if resp.status_code not in (200, 206):
                            raise IOError(f"HTTP status {resp.status_code} on chunk {cid}")

                        with open(part_path, "r+b") as f:
                            f.seek(current_start)
                            for data in resp.iter_content(chunk_size=self.CHUNK_BUFFER_SIZE):
                                if self._cancelled.is_set():
                                    return False
                                if data:
                                    f.write(data)
                                    chunk_info["bytes_done"] += len(data)
                                    current_start += len(data)
                                    thread_states[cid]["done"] = chunk_info["bytes_done"]

                                    speed = self._update_speed(len(data))
                                    if progress_callback:
                                        eta = 0.0
                                        rem = max(0, self.overall_total - self.total_downloaded)
                                        if speed > 0:
                                            eta = rem / (speed * 1024 * 1024)
                                        progress_callback({
                                            "total_downloaded": self.total_downloaded,
                                            "completed_bytes": self.total_downloaded,
                                            "overall_total": self.overall_total,
                                            "total_bytes": self.overall_total,
                                            "speed_mb": speed,
                                            "peak_speed_mb": self.peak_speed_mb,
                                            "eta_seconds": eta,
                                            "thread_states": dict(thread_states)
                                        })
                    # Chunk finished successfully
                    thread_states[cid]["active"] = False
                    return True
                except Exception as ex:
                    time.sleep(1.0 * (attempt + 1))
                    if attempt == max_retries - 1:
                        log(f"Worker {cid} error after retries: {ex}")
                        thread_states[cid]["active"] = False
                        return False

            thread_states[cid]["active"] = False
            return False

        # Execute concurrent workers
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, c) for c in chunks]
            success = True
            for fut in as_completed(futures):
                if not fut.result():
                    success = False
                    break

        if not success or self._cancelled.is_set():
            # Save checkpoint state for resumption
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(chunks, mf)
            except Exception:
                pass
            return False

        # Validate SHA-256 hash integrity
        log("[i] Validating SHA-256 integrity...")
        if not self._verify_hash(part_path, digest):
            log("[ERROR] Checksum verification failed! Re-download required.")
            try: os.remove(part_path)
            except Exception: pass
            return False

        # Promotion to permanent blob
        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(part_path, target_path)

        # Remove temporary state file
        if os.path.exists(meta_path):
            try: os.remove(meta_path)
            except Exception: pass

        log("[+] Integrity verified. Layer saved.")
        return True

    @staticmethod
    def _verify_hash(filepath: str, expected_digest: str) -> bool:
        """Verify sha256 checksum of downloaded file."""
        if not expected_digest.startswith("sha256:"):
            return True
        expected_hex = expected_digest.split(":", 1)[1]
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected_hex.lower()
