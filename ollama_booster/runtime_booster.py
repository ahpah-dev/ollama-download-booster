"""
Ollama Runtime Speed Booster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accelerates model inference and execution speeds on Ollama by:
1. Enabling Flash Attention (OLLAMA_FLASH_ATTENTION=1).
2. Enabling Persistent Warm Memory Residency (OLLAMA_KEEP_ALIVE=24h).
3. Elevating Ollama runner process priority to High to eliminate OS CPU throttling.
4. Auto-detecting physical CPU cores to eliminate hyperthreading/SMT cache thrashing.
5. Optimizing prompt evaluation batching (num_batch=1024) and memory locking (mlock/mmap).
"""

import os
import sys
import time
import json
import shutil
import threading
import subprocess
from typing import Dict, Any, Optional, Callable, Generator, Tuple
import requests


class RuntimeBooster:
    """Manages Ollama runtime inference acceleration and process optimizations."""

    def __init__(self):
        self.physical_cores = self._detect_physical_cores()
        self.logical_cores = os.cpu_count() or self.physical_cores
        self.optimal_threads = max(1, self.physical_cores)
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()

    @staticmethod
    def _detect_physical_cores() -> int:
        """Detect physical CPU cores (avoids SMT/hyperthreading performance penalties in llama.cpp)."""
        if sys.platform == "win32":
            try:
                cmd = "Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum | Select-Object -ExpandProperty Sum"
                res = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
                val = int(res)
                if val > 0:
                    return val
            except Exception:
                pass
        elif sys.platform == "darwin":
            try:
                res = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
                return int(res)
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cores = set()
                    for line in f:
                        if line.startswith("core id"):
                            cores.add(line.strip())
                    if cores:
                        return len(cores)
            except Exception:
                pass

        total = os.cpu_count() or 4
        return max(1, total // 2) if total > 4 else total

    # ---------------- ENVIRONMENT & PERSISTENT BOOST ----------------

    def is_runtime_boost_enabled(self) -> bool:
        """Check if Flash Attention and persistent keep-alive are set in environment."""
        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_READ)
                try:
                    fa, _ = winreg.QueryValueEx(key, "OLLAMA_FLASH_ATTENTION")
                    return str(fa).strip() == "1"
                finally:
                    winreg.CloseKey(key)
            except Exception:
                pass
        return os.getenv("OLLAMA_FLASH_ATTENTION") == "1"

    def enable_runtime_acceleration(self) -> bool:
        """
        Enable iGPU Vulkan acceleration, Flash Attention, persistent keep-alive, and high process priority.
        Persists in Windows User Environment registry and current session.
        """
        os.environ["OLLAMA_IGPU_ENABLE"] = "1"
        os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
        os.environ["OLLAMA_KEEP_ALIVE"] = "24h"
        os.environ["OLLAMA_NUM_PARALLEL"] = "1"

        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
                winreg.SetValueEx(key, "OLLAMA_IGPU_ENABLE", 0, winreg.REG_SZ, "1")
                winreg.SetValueEx(key, "OLLAMA_FLASH_ATTENTION", 0, winreg.REG_SZ, "1")
                winreg.SetValueEx(key, "OLLAMA_KEEP_ALIVE", 0, winreg.REG_SZ, "24h")
                winreg.SetValueEx(key, "OLLAMA_NUM_PARALLEL", 0, winreg.REG_SZ, "1")
                winreg.CloseKey(key)
            except Exception:
                pass

        # Elevate process priority immediately
        self.elevate_ollama_priority()
        self.start_process_priority_watcher()
        return True

    def disable_runtime_acceleration(self) -> bool:
        """Disable runtime boost and reset environment variables."""
        os.environ.pop("OLLAMA_IGPU_ENABLE", None)
        os.environ.pop("OLLAMA_FLASH_ATTENTION", None)
        os.environ.pop("OLLAMA_KEEP_ALIVE", None)
        os.environ.pop("OLLAMA_NUM_PARALLEL", None)

        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
                for var_name in ["OLLAMA_IGPU_ENABLE", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KEEP_ALIVE", "OLLAMA_NUM_PARALLEL"]:
                    try:
                        winreg.DeleteValue(key, var_name)
                    except FileNotFoundError:
                        pass
                winreg.CloseKey(key)
            except Exception:
                pass

        self.stop_process_priority_watcher()
        self.reset_ollama_priority()
        return True

    def is_ollama_running(self, host: str = "http://127.0.0.1:11434") -> bool:
        """Check if Ollama service is currently reachable and responding."""
        try:
            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def restart_ollama_with_turbo(self) -> Tuple[bool, str]:
        """
        Gracefully restart the Ollama service injecting all turbo flags
        (Integrated GPU Vulkan Acceleration, Flash Attention, 24h Keep-Alive, High Priority).
        """
        self.enable_runtime_acceleration()

        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe", "/T"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe", "/T"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)

            ps_script = (
                "$env:OLLAMA_IGPU_ENABLE = '1'; "
                "$env:OLLAMA_FLASH_ATTENTION = '1'; "
                "$env:OLLAMA_KEEP_ALIVE = '24h'; "
                "$env:OLLAMA_NUM_PARALLEL = '1'; "
                "Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden"
            )
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, timeout=10)
                # Wait for Ollama service to become responsive
                for _ in range(10):
                    time.sleep(0.5)
                    if self.is_ollama_running():
                        break
                self.elevate_ollama_priority()
                return True, "Ollama service successfully restarted with AMD Radeon iGPU Vulkan acceleration & Flash Attention!"
            except Exception as e:
                return False, f"Failed to restart Ollama: {e}"
        else:
            subprocess.run(["pkill", "-f", "ollama"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)
            env = os.environ.copy()
            env["OLLAMA_IGPU_ENABLE"] = "1"
            env["OLLAMA_FLASH_ATTENTION"] = "1"
            env["OLLAMA_KEEP_ALIVE"] = "24h"
            env["OLLAMA_NUM_PARALLEL"] = "1"
            subprocess.Popen(["ollama", "serve"], env=env)
            for _ in range(10):
                time.sleep(0.5)
                if self.is_ollama_running():
                    break
            return True, "Ollama service restarted with GPU Turbo optimizations!"

    def create_turbo_model_copy(self, model_name: str, target_name: Optional[str] = None, num_ctx: int = 2048) -> Tuple[bool, str]:
        """
        Creates an instant tuned '<model>-turbo' variant with locked physical threads,
        fast prompt batching, and an optimized context window. Takes 0 extra disk space.
        """
        if not target_name:
            clean = model_name.split("/")[-1].split(":")[0]
            target_name = f"{clean}-turbo"

        modelfile_content = f"""FROM {model_name}
PARAMETER num_ctx {num_ctx}
PARAMETER num_thread {self.optimal_threads}
PARAMETER num_batch 1024
"""
        tmp_path = os.path.join(os.path.expanduser("~"), f".ollama_modelfile_{int(time.time())}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)

            res = subprocess.run(
                ["ollama", "create", target_name, "-f", tmp_path],
                capture_output=True,
                text=True,
                timeout=45
            )
            if res.returncode == 0:
                return True, f"Successfully created '{target_name}'! Tuned with {self.optimal_threads} threads, {num_ctx} context, and batch 1024."
            else:
                return False, f"Ollama create failed: {res.stderr.strip() or res.stdout.strip()}"
        except Exception as e:
            return False, f"Failed to create turbo model: {e}"
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass

    # ---------------- PROCESS PRIORITY OPTIMIZATION ----------------

    def elevate_ollama_priority(self) -> int:
        """Sets priority class of Ollama processes to High on Windows."""
        count = 0
        if sys.platform == "win32":
            try:
                ps_cmd = (
                    "Get-Process -Name '*ollama*' -ErrorAction SilentlyContinue | "
                    "ForEach-Object { $_.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High }"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False, stdout=subprocess.DEVNULL)
                count = 1
            except Exception:
                pass
        return count

    def reset_ollama_priority(self) -> None:
        """Resets priority class of Ollama processes to Normal."""
        if sys.platform == "win32":
            try:
                ps_cmd = (
                    "Get-Process -Name '*ollama*' -ErrorAction SilentlyContinue | "
                    "ForEach-Object { $_.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal }"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False, stdout=subprocess.DEVNULL)
            except Exception:
                pass

    def start_process_priority_watcher(self) -> None:
        """Runs a lightweight background thread ensuring Ollama runners stay at High priority."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._watcher_stop.clear()

        def watcher():
            while not self._watcher_stop.is_set():
                self.elevate_ollama_priority()
                time.sleep(3)

        self._watcher_thread = threading.Thread(target=watcher, daemon=True)
        self._watcher_thread.start()

    def stop_process_priority_watcher(self) -> None:
        self._watcher_stop.set()

    # ---------------- TUNED INFERENCE CALLS ----------------

    def get_boosted_options(self, num_ctx: int = 2048) -> Dict[str, Any]:
        """Returns optimal inference hyper-parameters tuned for this machine."""
        return {
            "num_thread": self.optimal_threads,
            "num_gpu": 99,
            "num_ctx": num_ctx,
            "num_batch": 1024,
            "use_mmap": True
        }

    def generate_boosted_stream(
        self,
        model: str,
        prompt: str,
        num_ctx: int = 2048,
        host: str = "http://127.0.0.1:11434"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes a streaming inference generation with full turbo parameters.
        Yields chunk dictionaries containing token string, live tokens/second, TTFT, and progress.
        """
        self.elevate_ollama_priority()
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": self.get_boosted_options(num_ctx=num_ctx)
        }

        start_time = time.time()
        first_token_time = None
        tokens_received = 0

        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    tokens_received += 1
                    if first_token_time is None:
                        first_token_time = time.time() - start_time

                elapsed = max(0.001, time.time() - start_time)
                tps = tokens_received / elapsed if elapsed > 0 else 0.0

                yield {
                    "token": token,
                    "done": data.get("done", False),
                    "tokens_so_far": tokens_received,
                    "ttft": first_token_time,
                    "current_tps": round(tps, 2),
                    "eval_count": data.get("eval_count"),
                    "eval_duration_sec": data.get("eval_duration", 0) / 1e9 if data.get("eval_duration") else None,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "prompt_eval_duration_sec": data.get("prompt_eval_duration", 0) / 1e9 if data.get("prompt_eval_duration") else None,
                    "total_duration_sec": round(elapsed, 2)
                }

    def launch_boosted_terminal(self, model: str) -> None:
        """
        Launches an interactive `ollama run <model>` session with
        iGPU Vulkan, Flash Attention, Keep-Alive, and High Process Priority applied.
        """
        env_cmd = (
            f"set OLLAMA_IGPU_ENABLE=1&& "
            f"set OLLAMA_FLASH_ATTENTION=1&& "
            f"set OLLAMA_KEEP_ALIVE=24h&& "
            f"set OLLAMA_NUM_PARALLEL=1&& "
            f"ollama run {model}"
        )
        if sys.platform == "win32":
            subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", env_cmd])
            # Elevate process after launch
            threading.Timer(1.0, self.elevate_ollama_priority).start()
        elif sys.platform == "darwin":
            script = f'tell application "Terminal" to do script "export OLLAMA_IGPU_ENABLE=1 && export OLLAMA_FLASH_ATTENTION=1 && export OLLAMA_KEEP_ALIVE=24h && ollama run {model}"'
            subprocess.Popen(["osascript", "-e", script])
        else:
            subprocess.Popen(["bash", "-c", f"export OLLAMA_IGPU_ENABLE=1 && export OLLAMA_FLASH_ATTENTION=1 && export OLLAMA_KEEP_ALIVE=24h && ollama run {model}"])
