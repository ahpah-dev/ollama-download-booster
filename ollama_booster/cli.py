"""
Booster CLI Interface
~~~~~~~~~~~~~~~~~~~~~
Terminal commands for boosting Ollama model downloads, inspecting models,
running speed diagnostics, and starting the local booster proxy.
"""

import sys
import time
import argparse
from typing import Dict, Any

from .booster_engine import BoosterEngine
from .models_manager import OllamaManager
from .registry_client import RegistryClient
from .proxy_server import BoosterProxyServer

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Terminal styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    banner = f"""{CYAN}{BOLD}
   ____  __    __                         ____                   __           
  / __ \\/ /   / /___ _____ ___  ____ _   / __ )____  ____  _____/ /____  _____
 / / / / /   / / __ `/ __ `__ \\/ __ `/  / __  / __ \\/ __ \\/ ___/ __/ _ \\/ ___/
/ /_/ / /___/ / /_/ / / / / / / /_/ /  / /_/ / /_/ / /_/ (__  ) /_/  __/ /    
\\____/_____/_/\\__,_/_/ /_/ /_/\\__,_/  /_____/\\____/\\____/____/\\__/\\___/_/     
{RESET}{DIM}              High-Speed Multi-Stream Model Acceleration for Ollama{RESET}
"""
    print(banner)


def format_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def cmd_pull(args):
    print_banner()
    model = args.model
    threads = args.concurrency or 16

    print(f"{BOLD}Target Model:{RESET} {CYAN}{model}{RESET}")
    print(f"{BOLD}Booster State:{RESET} {GREEN}ACTIVE{RESET} ({threads} parallel streams)")
    print("-" * 65)

    manager = OllamaManager()
    engine = BoosterEngine(ollama_manager=manager)

    start_time = time.time()
    last_line_len = 0

    def on_progress(p: Dict[str, Any]):
        nonlocal last_line_len
        done = p.get("total_downloaded", 0)
        tot = p.get("overall_total", 0)
        speed = p.get("speed_mb", 0.0)
        eta = p.get("eta_seconds", 0.0)
        pct = (done / tot * 100) if tot > 0 else 0.0

        bar_len = 24
        filled = int(bar_len * (pct / 100))
        bar = f"{GREEN}{'=' * filled}{RESET}{DIM}{'-' * (bar_len - filled)}{RESET}"

        eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta//60)}m{int(eta%60)}s"
        status_line = (
            f"\r[{bar}] {pct:5.1f}% | "
            f"{format_bytes(done)}/{format_bytes(tot)} | "
            f"{CYAN}{speed:5.1f} MB/s{RESET} | "
            f"ETA: {eta_str}"
        )
        sys.stdout.write(status_line.ljust(last_line_len))
        sys.stdout.flush()
        last_line_len = max(last_line_len, len(status_line))

    def on_log(msg: str):
        nonlocal last_line_len
        sys.stdout.write("\r" + " " * last_line_len + "\r")
        print(f"  {msg}")
        sys.stdout.flush()

    try:
        success = engine.download_model(
            model_ref=model,
            concurrency=threads,
            progress_callback=on_progress,
            status_log_callback=on_log
        )
        print()  # Newline after progress bar
        print("-" * 65)

        elapsed = time.time() - start_time
        if success:
            avg_speed = (engine.total_downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            print(f"{GREEN}{BOLD}[*] Boosted Download Complete!{RESET}")
            print(f"• Total Transferred: {format_bytes(engine.total_downloaded)}")
            print(f"• Time Elapsed:      {elapsed:.1f} seconds")
            print(f"• Average Speed:     {CYAN}{avg_speed:.2f} MB/s{RESET}")
            print(f"• Peak Speed:        {CYAN}{engine.peak_speed_mb:.2f} MB/s{RESET}")
            print(f"\n{BOLD}To test your accelerated model instantly:{RESET}")
            print(f"  {CYAN}ollama run {model}{RESET}\n")
        else:
            print(f"{RED}[!] Download incomplete or stopped.{RESET}")
    except KeyboardInterrupt:
        engine.cancel()
        print(f"\n{YELLOW}[!] Aborted by user.{RESET}")


def cmd_list(args):
    print_banner()
    manager = OllamaManager()
    models = manager.list_installed_models()

    print(f"{BOLD}Locally Installed Ollama Models ({len(models)}):{RESET}")
    print(f"{DIM}Storage path: {manager.models_dir}{RESET}\n")

    if not models:
        print("  No models currently stored in local directory.")
        return

    print(f"{'NAME':<35} {'ID':<14} {'SIZE':<10} {'TAG':<10}")
    print("-" * 72)
    for m in models:
        print(f"{CYAN}{m['name']:<35}{RESET} {m['id']:<14} {format_bytes(m['size_bytes']):<10} {m['tag']:<10}")
    print("-" * 72)


def cmd_speed_test(args):
    """Diagnose network and show single-stream vs boosted multi-stream speed."""
    print_banner()
    print(f"{BOLD}Diagnostic Speed Benchmark{RESET}")
    print("Testing download throughput against Ollama model weights...")
    print("-" * 65)

    client = RegistryClient()
    # Use smollm:135m for fast benchmark
    manifest, _ = client.fetch_manifest("smollm:135m")
    layers = manifest.get("layers", [])
    model_layer = next((l for l in layers if "model" in l.get("mediaType", "")), layers[0])
    digest = model_layer["digest"]

    direct_url = client.resolve_blob_download_url("registry.ollama.ai", "library", "smollm", digest)

    test_bytes = 20 * 1024 * 1024  # 20 MB test sample
    session = client.session

    print("1. Measuring Standard Single-Stream Speed (Ollama default)...")
    t0 = time.time()
    r = session.get(direct_url, headers={"Range": f"bytes=0-{test_bytes-1}"}, stream=True, timeout=20)
    b_single = len(r.content)
    t_single = max(0.001, time.time() - t0)
    speed_single = (b_single / (1024 * 1024)) / t_single
    print(f"   [+] Single-Stream: {speed_single:.2f} MB/s ({t_single:.2f}s)")

    print("2. Measuring Boosted Multi-Stream Speed (16 parallel workers)...")
    import concurrent.futures
    streams = 16
    chunk_sz = test_bytes // streams
    def fetch_chunk(i):
        st = i * chunk_sz
        en = min(test_bytes - 1, (i + 1) * chunk_sz - 1)
        res = session.get(direct_url, headers={"Range": f"bytes={st}-{en}"}, timeout=20)
        return len(res.content)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as ex:
        chunks = list(ex.map(fetch_chunk, range(streams)))
    t_multi = max(0.001, time.time() - t0)
    b_multi = sum(chunks)
    speed_multi = (b_multi / (1024 * 1024)) / t_multi
    print(f"   [+] Multi-Stream:  {speed_multi:.2f} MB/s ({t_multi:.2f}s)")

    boost = speed_multi / speed_single if speed_single > 0 else 1.0
    print("-" * 65)
    print(f"{GREEN}{BOLD}[*] Boost Factor: {boost:.1f}x Faster!{RESET}")
    print(f"Enabling Ollama Download Booster cuts download times from {int(t_single*100)}s down to {int(t_multi*100)}s for a typical 2GB layer.\n")


def cmd_proxy(args):
    print_banner()
    port = args.port or 11435
    server = BoosterProxyServer(port=port)
    if not server.start():
        print(f"{RED}Port {port} is already in use or cannot be bound.{RESET}")
        return

    print(f"{GREEN}[+] Booster Local Proxy is RUNNING on http://127.0.0.1:{port}{RESET}")
    print(f"Endpoints available:")
    print(f"  GET  http://127.0.0.1:{port}/status")
    print(f"  GET  http://127.0.0.1:{port}/models")
    print(f"  POST http://127.0.0.1:{port}/boost  (JSON: {{'model': 'llama3.2', 'concurrency': 16}})")
    print(f"\nPress Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print(f"\n{YELLOW}Booster proxy stopped.{RESET}")


def cmd_boost_runtime(args):
    from .runtime_booster import RuntimeBooster
    print_banner()
    rb = RuntimeBooster()

    if args.action == "off":
        rb.disable_runtime_acceleration()
        print(f"{YELLOW}[*] Ollama Runtime Booster: DEACTIVATED{RESET}")
        print("  - Restored default Ollama environment variables.")
        print("  - Restored standard process priority.")
    else:
        rb.enable_runtime_acceleration()
        print(f"{GREEN}{BOLD}[*] Ollama Runtime Booster: ACTIVATED!{RESET}")
        print("  [+] Flash Attention enabled (OLLAMA_FLASH_ATTENTION=1)")
        print("  [+] Model Keep-Alive set to 24h (OLLAMA_KEEP_ALIVE=24h, zero cold-start reloads)")
        print(f"  [+] Physical CPU cores tuned: {rb.physical_cores} threads")
        print("  [+] Process priority elevated to High (0ms OS throttling)")
        print(f"\n{BOLD}Any new or running Ollama models will now run accelerated.{RESET}\n")


def cmd_run_boosted(args):
    from .runtime_booster import RuntimeBooster
    print_banner()
    rb = RuntimeBooster()
    model = args.model
    print(f"Launching {CYAN}{model}{RESET} with runtime acceleration...")
    rb.launch_boosted_terminal(model)


def cmd_bench_inference(args):
    from .runtime_booster import RuntimeBooster
    print_banner()
    rb = RuntimeBooster()
    model = args.model
    prompt = args.prompt or "Explain quantum computing in two concise sentences."

    print(f"{BOLD}Model:{RESET}   {CYAN}{model}{RESET}")
    print(f"{BOLD}Prompt:{RESET}  {prompt}")
    print(f"{BOLD}Hardware Tuning:{RESET} {rb.physical_cores} CPU Cores | Flash Attn: ON | High Priority")
    print("-" * 65)

    try:
        stream = rb.generate_boosted_stream(model, prompt)
        print(f"{CYAN}{BOLD}Generated Response:{RESET}\n")
        first_chunk = True
        final_chunk = None

        for chunk in stream:
            token = chunk.get("token", "")
            if token:
                sys.stdout.write(token)
                sys.stdout.flush()
            if chunk.get("done"):
                final_chunk = chunk

        print("\n" + "-" * 65)
        if final_chunk:
            tps = final_chunk.get("current_tps", 0.0)
            ttft = final_chunk.get("ttft", 0.0)
            toks = final_chunk.get("tokens_so_far", 0)
            dur = final_chunk.get("total_duration_sec", 0.0)
            print(f"{GREEN}{BOLD}[*] Inference Performance Stats:{RESET}")
            print(f"• Generation Speed:   {CYAN}{BOLD}{tps:.2f} tokens/second{RESET}")
            print(f"• Time-to-First-Token: {ttft:.2f}s")
            print(f"• Total Generated:    {toks} tokens in {dur:.2f}s")
            print(f"• Process Priority:   High (Active)")
            print(f"• Flash Attention:    Enabled\n")
    except Exception as e:
        print(f"\n{RED}[ERROR] Failed to run inference: {e}{RESET}")
        print("Ensure Ollama server is running (e.g. 'ollama serve').")


def cli_main():
    parser = argparse.ArgumentParser(
        prog="ollama-booster",
        description="Ollama Download & Runtime Booster - Accelerate downloads and model inference"
    )
    subparsers = parser.add_subparsers(dest="command")

    # pull
    pull_parser = subparsers.add_parser("pull", help="Boosted download of an Ollama model")
    pull_parser.add_argument("model", help="Model name (e.g. llama3.2, deepseek-r1:7b, qwen2.5:14b)")
    pull_parser.add_argument("-c", "--concurrency", type=int, default=16, help="Parallel stream count (default: 16)")

    # list
    subparsers.add_parser("list", help="List installed Ollama models")

    # benchmark
    subparsers.add_parser("test-speed", help="Run multi-stream vs single-stream download benchmark")

    # proxy
    proxy_parser = subparsers.add_parser("proxy", help="Run local accelerator proxy")
    proxy_parser.add_argument("-p", "--port", type=int, default=11435, help="Port to listen on (default: 11435)")

    # boost-runtime
    rt_parser = subparsers.add_parser("boost-runtime", help="Enable or disable persistent inference speed acceleration")
    rt_parser.add_argument("action", nargs="?", choices=["on", "off"], default="on", help="Enable (on) or disable (off)")

    # run-boosted
    run_parser = subparsers.add_parser("run-boosted", help="Launch an interactive model session with runtime acceleration")
    run_parser.add_argument("model", help="Model name to run")

    # bench-inference
    bench_parser = subparsers.add_parser("bench-inference", help="Benchmark generation speed (tokens/sec) of an installed model")
    bench_parser.add_argument("model", help="Model name to benchmark")
    bench_parser.add_argument("-p", "--prompt", help="Custom test prompt")

    args = parser.parse_args()

    if args.command == "pull":
        cmd_pull(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "test-speed":
        cmd_speed_test(args)
    elif args.command == "proxy":
        cmd_proxy(args)
    elif args.command == "boost-runtime":
        cmd_boost_runtime(args)
    elif args.command == "run-boosted":
        cmd_run_boosted(args)
    elif args.command == "bench-inference":
        cmd_bench_inference(args)
    else:
        # Default to GUI when no command is passed
        from .gui import start_gui
        start_gui()


if __name__ == "__main__":
    cli_main()
