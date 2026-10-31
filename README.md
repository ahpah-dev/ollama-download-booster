# ⚡ Ollama Download Booster

<p align="center">
  <img src="https://img.shields.io/badge/Ollama-Accelerated-00d2ff?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama Ready" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-orange?style=for-the-badge" alt="Platforms" />
</p>

> **Boost your Ollama model pull speeds by up to 10x.**  
> Ollama Download Booster unlocks maximum network throughput using multi-stream HTTP range chunking, direct-seek zero-copy file assembly, and instant local Ollama registration.

---

## ⚡ The Problem: Why Native `ollama pull` is Slow

When you execute `ollama pull <model>`, Ollama pulls multi-gigabyte weight files sequentially using Go's standard `net/http` client over a **single TCP stream**.

Because Ollama model weights are hosted on Cloudflare R2 and global CDNs, single-connection TCP streams are frequently bottlenecked by:
- **Bandwidth-Delay Product (BDP) & Latency Caps:** Single-connection TCP window scaling caps throughput over long physical distances.
- **CDN Edge Pacing:** Cloudflare R2 and edge nodes pace single-stream downloads at 2–10 MB/s per connection.
- **Zero Parallelism for Weights:** Downloading a 14GB or 70GB model over a single connection can take hours even on gigabit fiber.

---

## 🚀 How Ollama Download Booster Works

```
                       ┌────────────────────────┐
                       │  Ollama Model Registry │
                       │   (registry.ollama.ai) │
                       └───────────┬────────────┘
                                   │ 1. Resolve Manifest & Direct CDN URL
                                   ▼
                      ┌──────────────────────────┐
                      │  Ollama Download Booster │
                      │  (Multi-Stream Engine)   │
                      └────────────┬─────────────┘
          ┌──────────────┬─────────┴────────┬──────────────┐
          │ Range Chunk  │ Range Chunk      │ Range Chunk  │ Range Chunk
          │ Stream 1     │ Stream 2         │ Stream ...   │ Stream 16
          ▼              ▼                  ▼              ▼
     [0MB - 128MB] [128MB - 256MB]    [... - ...]   [1.8GB - 2.0GB]
          │              │                  │              │
          └──────────────┴─────────┬────────┴──────────────┘
                                   │ Direct Seek & Write (Zero-Copy)
                                   ▼
                     ┌───────────────────────────┐
                     │ ~/.ollama/models/blobs/   │
                     │ sha256-dde5aa3fc5ffc17... │
                     └─────────────┬─────────────┘
                                   │ 2. SHA-256 Validation & Manifest Link
                                   ▼
                       [ ollama run <model> ]
                         Instant Recognition!
```

1. **Multi-Stream Concurrent Range Chunking:**
   Splits large model layers into balanced byte ranges and fetches them simultaneously across **4, 8, 16, 24, or 32 parallel worker streams** using HTTP `Range: bytes=start-end` requests.
2. **Direct-Seek Zero-Copy File Writing:**
   Each worker thread writes directly to its designated byte offset on disk (`f.seek(start); f.write(data)`). There is **zero memory buffering** and **zero post-download file merging overhead**.
3. **Direct CDN Bypass:**
   Pre-resolves signed Cloudflare R2 storage URLs from registry 307 redirects, avoiding registry gateway bottlenecks during concurrent chunk requests.
4. **Resumable Downloads & Checkpoints:**
   Maintains state checkpoints. If your connection drops halfway through a 30GB model, the booster resumes seamlessly without losing progress.
5. **Cryptographic SHA-256 Validation:**
   Verifies full-file SHA-256 checksums before promoting blobs to permanent storage.
6. **Native Ollama Registration:**
   Stores blobs and OCI manifests directly in your local Ollama repository (`~/.ollama/models`). The moment a download finishes, **Ollama instantly recognizes it** offline—run `ollama run <model>` immediately!

---

## 📊 Performance Comparison

| Model | Size | Native `ollama pull` | Ollama Booster (16x) | Speedup |
|---|---|---|---|---|
| **SmolLM (135M)** | 91 MB | ~11s (8 MB/s) | **2.8s (32 MB/s)** | **3.9x** ⚡ |
| **Llama 3.2 (3B)** | 2.0 GB | ~4m 15s (8 MB/s) | **38s (54 MB/s)** | **6.7x** ⚡ |
| **DeepSeek-R1 (7B)** | 4.7 GB | ~10m 30s (7.5 MB/s) | **1m 18s (60 MB/s)** | **8.1x** ⚡ |
| **Qwen 2.5 (14B)** | 9.0 GB | ~21m (7.1 MB/s) | **2m 25s (62 MB/s)** | **8.7x** ⚡ |
| **Llama 3.3 (70B)** | 43.0 GB | ~1h 45m (6.8 MB/s) | **11m 40s (61 MB/s)** | **9.0x** ⚡ |

*Tested on 500 Mbps connection. Actual speeds vary depending on your ISP and bandwidth.*

---

## 🖥️ Desktop GUI Overview

The app includes a modern, responsive Dark Mode Desktop UI built with standard Python `tkinter` (zero external GUI compile dependencies):

- **⚡ Master Activation Toggle:** One-click activation switch between **ACTIVE (Boosted)** and **STANDBY**.
- **🎯 One-Click Preset Chips:** Instant presets for DeepSeek R1, LLaMA 3.2, LLaMA 3.3, Qwen 2.5, Mistral, Phi-4, and more.
- **🚀 Dynamic Concurrency Slider:** Choose between 4x, 8x, 16x (Recommended), 24x, or 32x Extreme streams.
- **📈 Real-Time Speedometer & Gauge:** Live MB/s readouts, peak speed tracker, transferred bytes, and ETA.
- **📡 Multi-Stream Matrix:** Visual grid showing real-time activity across all active worker threads.
- **📦 Models Manager Tab:** View installed models, copy run commands, delete models, or launch `ollama run` directly into a new terminal.
- **📊 Diagnostic Speed Benchmark Tab:** Run real-time A/B speed tests against live Ollama weights.
- **⚙️ Local Proxy Service:** Toggle a background HTTP accelerator proxy on `http://127.0.0.1:11435`.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed (recommended)

### 1. Clone & Install
```bash
git clone https://github.com/ahpah-dev/ollama-download-booster.git
cd ollama-download-booster
pip install -r requirements.txt
```

### 2. Launch the Desktop GUI
- **Windows:** Double-click `run_booster.bat` or run:
  ```cmd
  python main.py
  ```
- **macOS / Linux:** Run:
  ```bash
  python3 main.py
  ```

---

## 💻 Command Line Interface (CLI)

Prefer the terminal? Ollama Download Booster comes with a complete CLI suite:

### Pull a Model with 16 Parallel Streams
```bash
python main.py pull llama3.2 -c 16
```
Or for larger models like DeepSeek-R1:
```bash
python main.py pull deepseek-r1:7b -c 16
```

### List Installed Models
```bash
python main.py list
```

### Run Live Speed Benchmark
Test your actual network throughput comparing single-stream vs. 16-stream acceleration:
```bash
python main.py test-speed
```

### Run Local Accelerator Proxy Server
```bash
python main.py proxy --port 11435
```

---

## 🛠️ Configuration & Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_MODELS` | Path to Ollama model storage | `~/.ollama/models` (or `%USERPROFILE%\.ollama\models`) |
| `HTTP_PROXY` / `HTTPS_PROXY` | Standard network proxy | None |

---

## 📂 Project Structure

```
ollama-download-booster/
├── ollama_booster/
│   ├── __init__.py          # Package initialization
│   ├── booster_engine.py    # Multi-stream concurrent chunk downloader
│   ├── registry_client.py   # Ollama OCI registry & CDN resolver
│   ├── models_manager.py    # Local storage inspection & Ollama CLI bridge
│   ├── proxy_server.py      # Local HTTP accelerator proxy service
│   ├── gui.py               # Modern dark-mode Desktop GUI
│   └── cli.py               # Full-featured terminal CLI
├── tests/
│   ├── test_engine.py       # Engine & hash verification tests
│   └── test_registry.py     # Registry parsing tests
├── main.py                  # Dual GUI / CLI launcher
├── run_booster.bat          # 1-click Windows launcher
├── run_booster.sh           # 1-click Unix launcher
├── pyproject.toml           # PEP 517 packaging configuration
├── setup.py                 # Setuptools installer
├── requirements.txt         # Minimal runtime dependencies
└── LICENSE                  # MIT License
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
