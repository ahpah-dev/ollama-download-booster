"""
Ollama Download Booster - Desktop GUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modern dark-themed desktop interface for accelerated Ollama model downloads.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional

from .models_manager import OllamaManager
from .registry_client import RegistryClient
from .booster_engine import BoosterEngine
from .proxy_server import BoosterProxyServer
from .runtime_booster import RuntimeBooster


class BoosterApp(tk.Tk):
    """Main Application Window for Ollama Download Booster."""

    # UI Theme Palette
    COLOR_BG = "#0d1117"
    COLOR_CARD = "#161b22"
    COLOR_BORDER = "#30363d"
    COLOR_TEXT = "#e6edf3"
    COLOR_MUTED = "#8b949e"
    COLOR_CYAN = "#00d2ff"
    COLOR_ACCENT = "#58a6ff"
    COLOR_GREEN = "#2ea043"
    COLOR_GREEN_GLOW = "#3fb950"
    COLOR_RED = "#f85149"
    COLOR_AMBER = "#d29922"

    def __init__(self):
        super().__init__()
        self.title("Ollama Download & Runtime Booster ⚡")
        self.geometry("980x760")
        self.minsize(860, 660)
        self.configure(bg=self.COLOR_BG)

        # Business Logic Components
        self.manager = OllamaManager()
        self.engine = BoosterEngine(ollama_manager=self.manager)
        self.proxy_server = BoosterProxyServer(port=11435, manager=self.manager)
        self.runtime_booster = RuntimeBooster()

        # State Variables
        self.is_boosted = tk.BooleanVar(value=True)  # Download booster active by default
        self.is_runtime_boosted = tk.BooleanVar(value=self.runtime_booster.is_runtime_boost_enabled())
        if self.is_runtime_boosted.get():
            self.runtime_booster.start_process_priority_watcher()

        self.concurrency_var = tk.IntVar(value=16)
        self.model_input_var = tk.StringVar(value="llama3.2")
        self.is_downloading = False
        self.download_thread: Optional[threading.Thread] = None

        self._configure_styles()
        self._build_ui()
        self._refresh_installed_models()

    def _configure_styles(self):
        """Setup custom ttk dark styling."""
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        style.configure("TNotebook", background=self.COLOR_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.COLOR_CARD,
            foreground=self.COLOR_MUTED,
            padding=[16, 8],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.COLOR_BORDER)],
            foreground=[("selected", self.COLOR_CYAN)]
        )

        style.configure(
            "Booster.Horizontal.TProgressbar",
            troughcolor=self.COLOR_CARD,
            background=self.COLOR_GREEN_GLOW,
            borderwidth=0,
            thickness=14
        )

    def _build_ui(self):
        """Assemble all UI sections."""
        # Top Header & Activation Banner
        self._build_header()

        # Tabs Container
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 10))

        # Tab 1: Turbo Pull
        tab_pull = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(tab_pull, text="  ⚡ Turbo Pull  ")
        self._build_tab_pull(tab_pull)

        # Tab 2: Installed Models
        tab_models = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(tab_models, text="  📦 Models Manager  ")
        self._build_tab_models(tab_models)

        # Tab 3: Inference Turbo (Model Speed Booster)
        tab_inference = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(tab_inference, text="  🚀 Inference Turbo  ")
        self._build_tab_inference(tab_inference)

        # Tab 4: Speed Benchmark
        tab_benchmark = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(tab_benchmark, text="  📊 Speed Benchmark  ")
        self._build_tab_benchmark(tab_benchmark)

        # Tab 5: Booster Proxy & Settings
        tab_settings = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(tab_settings, text="  ⚙️ Settings & Proxy  ")
        self._build_tab_settings(tab_settings)

        # Bottom Status Bar
        self._build_status_bar()

    def _build_header(self):
        """Top banner with app branding and dual activation toggles."""
        header_frame = tk.Frame(self, bg=self.COLOR_BG)
        header_frame.pack(fill=tk.X, padx=20, pady=(15, 5))

        # Title & subtitle
        title_box = tk.Frame(header_frame, bg=self.COLOR_BG)
        title_box.pack(side=tk.LEFT)

        lbl_logo = tk.Label(
            title_box,
            text="⚡ OLLAMA DOWNLOAD & RUNTIME BOOSTER",
            font=("Segoe UI", 15, "bold"),
            fg=self.COLOR_CYAN,
            bg=self.COLOR_BG
        )
        lbl_logo.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text="Multi-Stream Download Acceleration + Hardware-Tuned Inference Engine",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_BG
        )
        lbl_sub.pack(anchor="w")

        # Dual Activation Cluster on Right
        actions_cluster = tk.Frame(header_frame, bg=self.COLOR_BG)
        actions_cluster.pack(side=tk.RIGHT)

        self.btn_activate = tk.Button(
            actions_cluster,
            text="⚡ DOWNLOADS: ACTIVE (16x)",
            font=("Segoe UI", 10, "bold"),
            bg="#1a472a",
            fg=self.COLOR_GREEN_GLOW,
            activebackground=self.COLOR_GREEN,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._toggle_booster
        )
        self.btn_activate.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_runtime_boost = tk.Button(
            actions_cluster,
            text="🚀 INFERENCE: ACTIVE" if self.is_runtime_boosted.get() else "○ INFERENCE: OFF",
            font=("Segoe UI", 10, "bold"),
            bg="#1b3a4b" if self.is_runtime_boosted.get() else "#262c36",
            fg=self.COLOR_CYAN if self.is_runtime_boosted.get() else self.COLOR_MUTED,
            activebackground="#204b63",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._toggle_runtime_booster
        )
        self.btn_runtime_boost.pack(side=tk.LEFT)

    def _toggle_booster(self):
        """Toggle download booster state between Active and Standby."""
        new_state = not self.is_boosted.get()
        self.is_boosted.set(new_state)
        self._update_booster_ui()

    def _update_booster_ui(self):
        if self.is_boosted.get():
            threads = self.concurrency_var.get()
            self.btn_activate.config(
                text=f"⚡ DOWNLOADS: ACTIVE ({threads}x)",
                bg="#1a472a",
                fg=self.COLOR_GREEN_GLOW
            )
            self._log_event(f"⚡ Download Booster Activated: {threads} parallel streams armed.")
        else:
            self.btn_activate.config(
                text="○ DOWNLOADS: STANDBY",
                bg="#262c36",
                fg=self.COLOR_MUTED
            )
            self._log_event("Download Booster switched to Standby mode (single connection).")

    def _toggle_runtime_booster(self):
        """Toggle Ollama inference speed booster."""
        new_val = not self.is_runtime_boosted.get()
        self.is_runtime_boosted.set(new_val)
        if new_val:
            self.runtime_booster.enable_runtime_acceleration()
            self.btn_runtime_boost.config(
                text="🚀 INFERENCE: ACTIVE",
                bg="#1b3a4b",
                fg=self.COLOR_CYAN
            )
            if hasattr(self, "btn_toggle_inf"):
                self.btn_toggle_inf.config(
                    text="⚡ INFERENCE BOOST: ACTIVE",
                    bg="#1a472a",
                    fg=self.COLOR_GREEN_GLOW
                )
            self._log_event("🚀 Inference Speed Booster ACTIVATED: Flash Attention enabled, Keep-Alive 24h, High Process Priority.")
            messagebox.showinfo(
                "Inference Booster Active",
                "🚀 Ollama Model Speed Booster is now ACTIVE!\n\n"
                "• Flash Attention enabled (OLLAMA_FLASH_ATTENTION=1)\n"
                "• Persistent Warm VRAM Residency (OLLAMA_KEEP_ALIVE=24h, zero reload delay)\n"
                f"• CPU Physical Core Affinity: {self.runtime_booster.physical_cores} threads\n"
                "• Process Priority elevated to High (0ms OS thread delay)\n\n"
                "All models will now run significantly faster!"
            )
        else:
            self.runtime_booster.disable_runtime_acceleration()
            self.btn_runtime_boost.config(
                text="○ INFERENCE: OFF",
                bg="#262c36",
                fg=self.COLOR_MUTED
            )
            if hasattr(self, "btn_toggle_inf"):
                self.btn_toggle_inf.config(
                    text="○ INFERENCE BOOST: OFF",
                    bg="#262c36",
                    fg=self.COLOR_MUTED
                )
            self._log_event("Inference Speed Booster deactivated. Restored default Ollama settings.")

    # ---------------- TAB 1: TURBO PULL ----------------

    def _build_tab_pull(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input Card
        input_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        input_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            input_card,
            text="MODEL NAME OR TAG",
            font=("Segoe UI", 8, "bold"),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=15, pady=(12, 4))

        entry_row = tk.Frame(input_card, bg=self.COLOR_CARD)
        entry_row.pack(fill=tk.X, padx=15, pady=(0, 8))

        self.model_entry = tk.Entry(
            entry_row,
            textvariable=self.model_input_var,
            font=("Consolas", 12),
            bg="#0d1117",
            fg=self.COLOR_TEXT,
            insertbackground=self.COLOR_CYAN,
            relief=tk.FLAT,
            bd=5
        )
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.btn_start = tk.Button(
            entry_row,
            text="⚡ Boost & Pull",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_GREEN,
            fg="#ffffff",
            activebackground=self.COLOR_GREEN_GLOW,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            cursor="hand2",
            command=self._start_download
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_cancel = tk.Button(
            entry_row,
            text="⏹ Stop",
            font=("Segoe UI", 10, "bold"),
            bg="#21262d",
            fg=self.COLOR_RED,
            activebackground=self.COLOR_RED,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            state=tk.DISABLED,
            cursor="hand2",
            command=self._stop_download
        )
        self.btn_cancel.pack(side=tk.LEFT)

        # Quick Preset Chips
        presets_frame = tk.Frame(input_card, bg=self.COLOR_CARD)
        presets_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        tk.Label(
            presets_frame,
            text="Popular Presets:",
            font=("Segoe UI", 8),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT, padx=(0, 6))

        presets = [
            ("DeepSeek R1 (7B)", "deepseek-r1:7b"),
            ("Llama 3.2 (3B)", "llama3.2:3b"),
            ("Qwen 2.5 (7B)", "qwen2.5:7b"),
            ("Mistral (7B)", "mistral"),
            ("SmolLM (135M)", "smollm:135m"),
            ("🤗 HF: Spark 4B", "hf.co/SC117/Spark-X2.6-4B-abliterated-FIT-GGUF:Q4_K_M")
        ]

        for label, tag in presets:
            btn = tk.Button(
                presets_frame,
                text=label,
                font=("Segoe UI", 8),
                bg="#21262d",
                fg=self.COLOR_TEXT,
                activebackground=self.COLOR_BORDER,
                activeforeground=self.COLOR_CYAN,
                relief=tk.FLAT,
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2",
                command=lambda t=tag: self._select_preset(t)
            )
            btn.pack(side=tk.LEFT, padx=3)

        # Tip label for Hugging Face support
        tk.Label(
            input_card,
            text="💡 Direct Hugging Face support: Enter 'hf.co/username/repository:quant' to pull GGUFs accelerated.",
            font=("Segoe UI", 8),
            fg="#79c0ff",
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=15, pady=(0, 6))

        # Concurrency selector row
        concur_row = tk.Frame(input_card, bg=self.COLOR_CARD)
        concur_row.pack(fill=tk.X, padx=15, pady=(4, 12))

        tk.Label(
            concur_row,
            text="Acceleration Streams:",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT, padx=(0, 10))

        for c_val, c_lbl in [(4, "4x"), (8, "8x"), (16, "16x (Recommended)"), (24, "24x"), (32, "32x (Extreme)")]:
            rb = tk.Radiobutton(
                concur_row,
                text=c_lbl,
                value=c_val,
                variable=self.concurrency_var,
                font=("Segoe UI", 9),
                fg=self.COLOR_TEXT,
                bg=self.COLOR_CARD,
                selectcolor=self.COLOR_BG,
                activebackground=self.COLOR_CARD,
                activeforeground=self.COLOR_CYAN,
                command=self._update_booster_ui
            )
            rb.pack(side=tk.LEFT, padx=6)

        # Metrics & Gauge Dashboard
        dash_frame = tk.Frame(container, bg=self.COLOR_BG)
        dash_frame.pack(fill=tk.X, pady=(0, 10))

        # Speedometer Card (Left)
        card_speed = tk.Frame(dash_frame, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        card_speed.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(card_speed, text="LIVE SPEED", font=("Segoe UI", 8, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_speed_val = tk.Label(card_speed, text="0.0 MB/s", font=("Consolas", 22, "bold"), fg=self.COLOR_CYAN, bg=self.COLOR_CARD)
        self.lbl_speed_val.pack(anchor="w", padx=12, pady=(0, 2))
        self.lbl_speed_sub = tk.Label(card_speed, text="Peak: 0.0 MB/s  |  Booster: Armed", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg=self.COLOR_CARD)
        self.lbl_speed_sub.pack(anchor="w", padx=12, pady=(0, 8))

        # Progress Stats Card (Right)
        card_progress = tk.Frame(dash_frame, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        card_progress.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(card_progress, text="TRANSFERRED & ETA", font=("Segoe UI", 8, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_transferred_val = tk.Label(card_progress, text="0 MB / 0 MB (0%)", font=("Consolas", 15, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_CARD)
        self.lbl_transferred_val.pack(anchor="w", padx=12, pady=(0, 2))
        self.lbl_eta_val = tk.Label(card_progress, text="ETA: --", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg=self.COLOR_CARD)
        self.lbl_eta_val.pack(anchor="w", padx=12, pady=(0, 8))

        # Overall Progress Bar
        self.prog_bar = ttk.Progressbar(container, style="Booster.Horizontal.TProgressbar", mode="determinate")
        self.prog_bar.pack(fill=tk.X, pady=(0, 10))

        # Multi-Stream Visual Matrix Card
        stream_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        stream_card.pack(fill=tk.X, pady=(0, 10))

        stream_hdr = tk.Frame(stream_card, bg=self.COLOR_CARD)
        stream_hdr.pack(fill=tk.X, padx=12, pady=(8, 4))
        tk.Label(stream_hdr, text="ACTIVE PARALLEL STREAMS", font=("Segoe UI", 8, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(side=tk.LEFT)
        self.lbl_stream_status = tk.Label(stream_hdr, text="Streams Idle", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg=self.COLOR_CARD)
        self.lbl_stream_status.pack(side=tk.RIGHT)

        self.stream_canvas = tk.Canvas(stream_card, height=36, bg=self.COLOR_CARD, highlightthickness=0)
        self.stream_canvas.pack(fill=tk.X, padx=12, pady=(0, 8))
        self._draw_stream_slots(0)

        # Log Terminal Card
        log_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        log_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(log_card, text="ACTIVITY LOG", font=("Segoe UI", 8, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=12, pady=(8, 4))

        self.log_text = tk.Text(
            log_card,
            bg="#0d1117",
            fg="#c9d1d9",
            font=("Consolas", 9),
            relief=tk.FLAT,
            height=7,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

    def _draw_stream_slots(self, active_count: int, thread_states: Optional[Dict[int, Any]] = None):
        """Draw interactive visual bars for each parallel connection stream."""
        self.stream_canvas.delete("all")
        width = self.stream_canvas.winfo_width()
        if width <= 10:
            width = 880
        total_slots = self.concurrency_var.get()
        slot_w = (width - 24) / total_slots
        gap = 3

        for i in range(total_slots):
            x0 = 12 + i * slot_w
            x1 = x0 + slot_w - gap
            y0 = 6
            y1 = 28

            is_active = False
            pct = 0.0
            if thread_states and i in thread_states:
                st = thread_states[i]
                is_active = st.get("active", False)
                tot = st.get("total", 1)
                done = st.get("done", 0)
                pct = min(1.0, done / tot) if tot > 0 else 0.0

            bg_color = "#21262d"
            fill_color = self.COLOR_GREEN_GLOW if is_active else "#30363d"

            self.stream_canvas.create_rectangle(x0, y0, x1, y1, fill=bg_color, outline="#30363d", width=1)
            if pct > 0:
                inner_h = (y1 - y0) * pct
                self.stream_canvas.create_rectangle(x0 + 1, y1 - inner_h, x1 - 1, y1 - 1, fill=fill_color, width=0)

    def _select_preset(self, tag: str):
        self.model_input_var.set(tag)
        self._log_event(f"Selected preset model: {tag}")

    def _log_event(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log_text.see(tk.END)

    def _start_download(self):
        model = self.model_input_var.get().strip()
        if not model:
            messagebox.showwarning("Input Required", "Please enter a valid model name or tag.")
            return

        concurrency = self.concurrency_var.get() if self.is_boosted.get() else 1
        self.is_downloading = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.prog_bar["value"] = 0

        self._log_event(f"Initiating boosted pull for '{model}' (streams: {concurrency})...")

        def run():
            def on_progress(p):
                self.after(0, lambda: self._update_progress_ui(p))

            def on_log(msg):
                self.after(0, lambda: self._log_event(msg))

            try:
                success = self.engine.download_model(
                    model_ref=model,
                    concurrency=concurrency,
                    progress_callback=on_progress,
                    status_log_callback=on_log
                )
                self.after(0, lambda: self._on_download_complete(success, model))
            except Exception as e:
                self.after(0, lambda: self._on_download_error(str(e)))

        self.download_thread = threading.Thread(target=run, daemon=True)
        self.download_thread.start()

    def _stop_download(self):
        if self.is_downloading:
            self._log_event("Cancelling download...")
            self.engine.cancel()
            self.btn_cancel.config(state=tk.DISABLED)

    def _update_progress_ui(self, p: Dict[str, Any]):
        done = p.get("total_downloaded", 0)
        tot = p.get("overall_total", 0)
        speed = p.get("speed_mb", 0.0)
        peak = p.get("peak_speed_mb", 0.0)
        eta = p.get("eta_seconds", 0.0)
        threads = p.get("thread_states", {})

        pct = (done / tot * 100) if tot > 0 else 0.0
        self.prog_bar["value"] = pct

        self.lbl_speed_val.config(text=f"{speed:.1f} MB/s")
        self.lbl_speed_sub.config(text=f"Peak: {peak:.1f} MB/s  |  Booster: Active ({self.concurrency_var.get()}x)")

        done_mb = done / (1024 * 1024)
        tot_mb = tot / (1024 * 1024)
        self.lbl_transferred_val.config(text=f"{done_mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)")

        eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta//60)}m {int(eta%60)}s"
        self.lbl_eta_val.config(text=f"ETA: {eta_str}")

        active_count = sum(1 for t in threads.values() if t.get("active"))
        self.lbl_stream_status.config(text=f"{active_count}/{len(threads)} Active Streams")
        self._draw_stream_slots(active_count, threads)

    def _on_download_complete(self, success: bool, model: str):
        self.is_downloading = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.lbl_stream_status.config(text="Streams Idle")
        self._draw_stream_slots(0)

        if success:
            self.prog_bar["value"] = 100
            self._refresh_installed_models()
            messagebox.showinfo(
                "Download Complete",
                f"Model '{model}' downloaded and registered successfully!\n\n"
                f"You can now run it immediately with:\nollama run {model}"
            )
        else:
            if not self.engine.is_cancelled():
                messagebox.showerror("Download Incomplete", f"Failed to complete download of '{model}'.")

    def _on_download_error(self, err_msg: str):
        self.is_downloading = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self._log_event(f"❌ Error: {err_msg}")
        messagebox.showerror("Error", err_msg)

    # ---------------- TAB 2: INSTALLED MODELS ----------------

    def _build_tab_models(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Toolbar
        bar = tk.Frame(container, bg=self.COLOR_BG)
        bar.pack(fill=tk.X, pady=(0, 10))

        btn_run_boosted = tk.Button(
            bar,
            text="⚡ Run Boosted",
            font=("Segoe UI", 9, "bold"),
            bg="#1b3a4b",
            fg=self.COLOR_CYAN,
            activebackground="#204b63",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._run_selected_model_boosted
        )
        btn_run_boosted.pack(side=tk.LEFT, padx=(0, 6))

        btn_turbo_profile = tk.Button(
            bar,
            text="⚡ Create -turbo Copy",
            font=("Segoe UI", 9),
            bg=self.COLOR_CARD,
            fg=self.COLOR_CYAN,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._create_selected_model_turbo
        )
        btn_turbo_profile.pack(side=tk.LEFT, padx=(0, 6))

        btn_run = tk.Button(
            bar,
            text="🚀 Standard Run",
            font=("Segoe UI", 9),
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._run_selected_model
        )
        btn_run.pack(side=tk.LEFT, padx=(0, 6))

        btn_copy = tk.Button(
            bar,
            text="📋 Copy Run Command",
            font=("Segoe UI", 9),
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._copy_run_cmd
        )
        btn_copy.pack(side=tk.LEFT, padx=(0, 6))

        btn_del = tk.Button(
            bar,
            text="🗑 Delete",
            font=("Segoe UI", 9),
            bg=self.COLOR_CARD,
            fg=self.COLOR_RED,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._delete_selected_model
        )
        btn_del.pack(side=tk.LEFT, padx=(0, 6))

        btn_refresh = tk.Button(
            bar,
            text="🔄 Refresh",
            font=("Segoe UI", 9),
            bg=self.COLOR_CARD,
            fg=self.COLOR_MUTED,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._refresh_installed_models
        )
        btn_refresh.pack(side=tk.RIGHT)

        # Models Treeview
        tree_frame = tk.Frame(container, bg=self.COLOR_CARD)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "id", "size", "tag")
        self.tree_models = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        self.tree_models.heading("name", text="MODEL NAME", anchor="w")
        self.tree_models.heading("id", text="ID", anchor="w")
        self.tree_models.heading("size", text="SIZE", anchor="w")
        self.tree_models.heading("tag", text="TAG", anchor="w")

        self.tree_models.column("name", width=340)
        self.tree_models.column("id", width=140)
        self.tree_models.column("size", width=120)
        self.tree_models.column("tag", width=100)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_models.yview)
        self.tree_models.configure(yscrollcommand=scrollbar.set)

        self.tree_models.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh_installed_models(self):
        for item in self.tree_models.get_children():
            self.tree_models.delete(item)

        models = self.manager.list_installed_models()
        model_names = []
        for m in models:
            model_names.append(m["name"])
            size_str = f"{m['size_gb']} GB" if m['size_gb'] >= 1.0 else f"{m['size_mb']} MB"
            self.tree_models.insert("", tk.END, values=(m["name"], m["id"], size_str, m["tag"]))

        if hasattr(self, "inf_model_cb"):
            self.inf_model_cb["values"] = model_names
            if model_names and not self.inf_model_cb.get():
                self.inf_model_cb.set(model_names[0])

    def _get_selected_model_name(self) -> Optional[str]:
        sel = self.tree_models.selection()
        if not sel:
            messagebox.showinfo("Selection", "Please select a model from the list.")
            return None
        values = self.tree_models.item(sel[0], "values")
        return values[0] if values else None

    def _run_selected_model(self):
        name = self._get_selected_model_name()
        if name:
            self.manager.run_model_terminal(name)

    def _run_selected_model_boosted(self):
        name = self._get_selected_model_name()
        if name:
            self.runtime_booster.launch_boosted_terminal(name)

    def _copy_run_cmd(self):
        name = self._get_selected_model_name()
        if name:
            cmd = f"ollama run {name}"
            self.clipboard_clear()
            self.clipboard_append(cmd)
            messagebox.showinfo("Copied", f"Copied to clipboard:\n{cmd}")

    def _delete_selected_model(self):
        name = self._get_selected_model_name()
        if name and messagebox.askyesno("Confirm Delete", f"Delete model '{name}' from local Ollama storage?"):
            ok, msg = self.manager.delete_model(name)
            if ok:
                self._refresh_installed_models()
                messagebox.showinfo("Deleted", msg)
            else:
                messagebox.showerror("Error", msg)

    def _create_selected_model_turbo(self, target_model: Optional[str] = None):
        name = target_model or self._get_selected_model_name()
        if not name:
            return
        if "-turbo" in name:
            messagebox.showinfo("Already Tuned", f"'{name}' is already an optimized turbo profile.")
            return

        ctx_val = 2048
        if hasattr(self, "inf_ctx_var"):
            try:
                ctx_val = int(self.inf_ctx_var.get())
            except Exception:
                pass

        def worker():
            ok, msg = self.runtime_booster.create_turbo_model_copy(name, num_ctx=ctx_val)
            def finish():
                if ok:
                    self._refresh_installed_models()
                    messagebox.showinfo("Turbo Profile Created", msg)
                else:
                    messagebox.showerror("Turbo Profile Error", msg)
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _restart_ollama_turbo(self):
        if messagebox.askyesno(
            "Restart Ollama with Turbo",
            "This will restart the Ollama service injecting:\n\n"
            "• AMD Radeon iGPU Vulkan Acceleration (OLLAMA_IGPU_ENABLE=1)\n"
            "• Flash Attention (OLLAMA_FLASH_ATTENTION=1)\n"
            "• 24h Keep-Alive Residency (OLLAMA_KEEP_ALIVE=24h)\n"
            "• Windows High Process Priority\n\n"
            "Restart Ollama service now?"
        ):
            if hasattr(self, "btn_restart_ollama"):
                self.btn_restart_ollama.config(state=tk.DISABLED, text="⏳ Restarting...")
            def worker():
                ok, msg = self.runtime_booster.restart_ollama_with_turbo()
                def finish():
                    if hasattr(self, "btn_restart_ollama"):
                        self.btn_restart_ollama.config(state=tk.NORMAL, text="🔄 Apply & Restart Ollama with Turbo")
                    self._update_ollama_service_indicator()
                    if ok:
                        messagebox.showinfo("Ollama Turbo Active", msg)
                    else:
                        messagebox.showerror("Restart Error", msg)
                self.after(0, finish)
            threading.Thread(target=worker, daemon=True).start()

    def _update_ollama_service_indicator(self):
        running = self.runtime_booster.is_ollama_running()
        if hasattr(self, "lbl_ollama_service_status"):
            if running:
                self.lbl_ollama_service_status.config(
                    text="● Ollama Service: Active (127.0.0.1:11434 - Vulkan Turbo)",
                    fg=self.COLOR_GREEN_GLOW
                )
            else:
                self.lbl_ollama_service_status.config(
                    text="○ Ollama Service: Offline",
                    fg=self.COLOR_RED
                )

    def _quick_pull_model(self, model_name: str):
        if hasattr(self, "notebook"):
            self.notebook.select(0)
        self.model_input_var.set(model_name)
        if not self.is_downloading:
            self._start_download()

    # ---------------- TAB 3: INFERENCE TURBO (SPEED BOOSTER) ----------------

    def _build_tab_inference(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. Master Toggle & Acceleration Status Card
        top_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        top_card.pack(fill=tk.X, pady=(0, 10))

        top_header_row = tk.Frame(top_card, bg=self.COLOR_CARD)
        top_header_row.pack(fill=tk.X, padx=15, pady=(10, 4))

        title_col = tk.Frame(top_header_row, bg=self.COLOR_CARD)
        title_col.pack(side=tk.LEFT)

        tk.Label(
            title_col,
            text="⚡ INFERENCE TURBO ENGINE",
            font=("Segoe UI", 11, "bold"),
            fg=self.COLOR_CYAN,
            bg=self.COLOR_CARD
        ).pack(anchor="w")

        tk.Label(
            title_col,
            text="AMD iGPU Vulkan acceleration, Flash Attention, 24h VRAM residency & physical core optimization",
            font=("Segoe UI", 8),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(anchor="w")

        self.btn_toggle_inf = tk.Button(
            top_header_row,
            text="⚡ INFERENCE BOOST: ACTIVE" if self.is_runtime_boosted.get() else "○ INFERENCE BOOST: OFF",
            font=("Segoe UI", 10, "bold"),
            bg="#1a472a" if self.is_runtime_boosted.get() else "#262c36",
            fg=self.COLOR_GREEN_GLOW if self.is_runtime_boosted.get() else self.COLOR_MUTED,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._toggle_runtime_booster
        )
        self.btn_toggle_inf.pack(side=tk.RIGHT)

        # 5 Feature Badges Row
        badges_frame = tk.Frame(top_card, bg=self.COLOR_CARD)
        badges_frame.pack(fill=tk.X, padx=15, pady=(4, 8))

        features = [
            ("⚡ AMD iGPU Vulkan", "OLLAMA_IGPU_ENABLE=1 (Vulkan offload)"),
            ("⚡ Flash Attention", "OLLAMA_FLASH_ATTENTION=1 (Fast generation)"),
            ("⚡ 24h Keep-Alive", "Zero cold-start 5-min reloads"),
            ("⚡ Core Affinity", f"{self.runtime_booster.physical_cores} Physical Cores (No SMT thrash)"),
            ("⚡ High Priority", "0ms Windows thread throttling")
        ]

        for label, sub in features:
            badge = tk.Frame(badges_frame, bg="#21262d", highlightbackground="#30363d", highlightthickness=1)
            badge.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(badge, text=label, font=("Segoe UI", 8, "bold"), fg=self.COLOR_TEXT, bg="#21262d").pack(anchor="w", padx=6, pady=(4, 1))
            tk.Label(badge, text=sub, font=("Segoe UI", 7), fg=self.COLOR_MUTED, bg="#21262d").pack(anchor="w", padx=6, pady=(0, 4))

        # Service Action & Restart Row
        svc_bar = tk.Frame(top_card, bg="#161b22", highlightbackground="#30363d", highlightthickness=1)
        svc_bar.pack(fill=tk.X, padx=15, pady=(0, 10))

        svc_left = tk.Frame(svc_bar, bg="#161b22")
        svc_left.pack(side=tk.LEFT, padx=10, pady=6)

        is_running = self.runtime_booster.is_ollama_running()
        self.lbl_ollama_service_status = tk.Label(
            svc_left,
            text="● Ollama Service: Active (127.0.0.1:11434 - Vulkan Turbo)" if is_running else "○ Ollama Service: Offline",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_GREEN_GLOW if is_running else self.COLOR_RED,
            bg="#161b22"
        )
        self.lbl_ollama_service_status.pack(anchor="w")

        tk.Label(
            svc_left,
            text="💡 Apply Turbo: Ollama only initializes GPU Vulkan & Flash Attention on service startup.",
            font=("Segoe UI", 7),
            fg=self.COLOR_MUTED,
            bg="#161b22"
        ).pack(anchor="w")

        self.btn_restart_ollama = tk.Button(
            svc_bar,
            text="🔄 Apply & Restart Ollama with Turbo",
            font=("Segoe UI", 9, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._restart_ollama_turbo
        )
        self.btn_restart_ollama.pack(side=tk.RIGHT, padx=10, pady=6)

        # 2. Interactive Testing & Speedometer Card
        test_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        test_card.pack(fill=tk.BOTH, expand=True)

        test_header = tk.Frame(test_card, bg=self.COLOR_CARD)
        test_header.pack(fill=tk.X, padx=15, pady=(8, 4))

        tk.Label(
            test_header,
            text="LIVE MODEL INFERENCE BENCHMARK & SPEEDOMETER",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT)

        # Model Selection & Prompt Input Row
        input_row = tk.Frame(test_card, bg=self.COLOR_CARD)
        input_row.pack(fill=tk.X, padx=15, pady=(0, 6))

        tk.Label(input_row, text="Model:", font=("Segoe UI", 9, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_CARD).pack(side=tk.LEFT, padx=(0, 6))
        self.inf_model_cb = ttk.Combobox(input_row, font=("Consolas", 10), state="readonly", width=26)
        self.inf_model_cb.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(input_row, text="Ctx:", font=("Segoe UI", 8, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(side=tk.LEFT, padx=(0, 3))
        self.inf_ctx_var = tk.StringVar(value="2048")
        ctx_cb = ttk.Combobox(input_row, textvariable=self.inf_ctx_var, values=["1024", "2048", "4096", "8192"], width=5, state="readonly")
        ctx_cb.pack(side=tk.LEFT, padx=(0, 8))

        btn_turbo_profile = tk.Button(
            input_row,
            text="⚡ Create -turbo Copy",
            font=("Segoe UI", 8, "bold"),
            bg="#1f3a4d",
            fg=self.COLOR_CYAN,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=lambda: self._create_selected_model_turbo(self.inf_model_cb.get().strip())
        )
        btn_turbo_profile.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_inf_run = tk.Button(
            input_row,
            text="⚡ Stream & Measure Speed",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_GREEN,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=3,
            cursor="hand2",
            command=self._start_inference_test
        )
        self.btn_inf_run.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_inf_terminal = tk.Button(
            input_row,
            text="🚀 Terminal",
            font=("Segoe UI", 9),
            bg="#21262d",
            fg=self.COLOR_CYAN,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._launch_inference_terminal
        )
        self.btn_inf_terminal.pack(side=tk.LEFT)

        # Prompt Row with Quick Preset Buttons
        prompt_box = tk.Frame(test_card, bg=self.COLOR_CARD)
        prompt_box.pack(fill=tk.X, padx=15, pady=(0, 6))

        self.inf_prompt_var = tk.StringVar(value="Explain artificial intelligence in two concise sentences.")
        self.inf_prompt_entry = tk.Entry(
            prompt_box,
            textvariable=self.inf_prompt_var,
            font=("Consolas", 10),
            bg="#0d1117",
            fg=self.COLOR_TEXT,
            insertbackground=self.COLOR_CYAN,
            relief=tk.FLAT,
            bd=3
        )
        self.inf_prompt_entry.pack(fill=tk.X, pady=(0, 4))

        presets_row = tk.Frame(prompt_box, bg=self.COLOR_CARD)
        presets_row.pack(fill=tk.X)

        tk.Label(presets_row, text="Quick Prompts:", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(side=tk.LEFT, padx=(0, 6))
        for q_lbl, q_text in [
            ("Short Fact", "Give me 1 fascinating fact about astronomy."),
            ("Python Code", "Write a Python function to check for palindromes."),
            ("Speed Benchmark", "Write a 50-word poem about speed and light.")
        ]:
            btn_q = tk.Button(
                presets_row,
                text=q_lbl,
                font=("Segoe UI", 8),
                bg="#21262d",
                fg=self.COLOR_TEXT,
                relief=tk.FLAT,
                padx=8,
                pady=1,
                cursor="hand2",
                command=lambda t=q_text: self.inf_prompt_var.set(t)
            )
            btn_q.pack(side=tk.LEFT, padx=3)

        # Digital Gauges Row
        gauge_row = tk.Frame(test_card, bg=self.COLOR_CARD)
        gauge_row.pack(fill=tk.X, padx=15, pady=(2, 6))

        # Gauge 1: Generation Speed
        g1 = tk.Frame(gauge_row, bg="#0d1117", highlightbackground="#30363d", highlightthickness=1)
        g1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(g1, text="GENERATION SPEED", font=("Segoe UI", 7, "bold"), fg=self.COLOR_MUTED, bg="#0d1117").pack(anchor="w", padx=10, pady=(4, 0))
        self.lbl_inf_tps = tk.Label(g1, text="0.0 tok/s", font=("Consolas", 16, "bold"), fg=self.COLOR_CYAN, bg="#0d1117")
        self.lbl_inf_tps.pack(anchor="w", padx=10, pady=(0, 2))
        self.lbl_inf_ttft = tk.Label(g1, text="TTFT: --  |  Prompt Eval: --", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg="#0d1117")
        self.lbl_inf_ttft.pack(anchor="w", padx=10, pady=(0, 4))

        # Gauge 2: Tokens & Latency
        g2 = tk.Frame(gauge_row, bg="#0d1117", highlightbackground="#30363d", highlightthickness=1)
        g2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(g2, text="TOKENS & LATENCY", font=("Segoe UI", 7, "bold"), fg=self.COLOR_MUTED, bg="#0d1117").pack(anchor="w", padx=10, pady=(4, 0))
        self.lbl_inf_tokens = tk.Label(g2, text="0 tokens", font=("Consolas", 16, "bold"), fg=self.COLOR_TEXT, bg="#0d1117")
        self.lbl_inf_tokens.pack(anchor="w", padx=10, pady=(0, 2))
        self.lbl_inf_dur = tk.Label(g2, text="Total Duration: 0.0s", font=("Segoe UI", 8), fg=self.COLOR_MUTED, bg="#0d1117")
        self.lbl_inf_dur.pack(anchor="w", padx=10, pady=(0, 4))

        # Streaming Text Output Box
        out_frame = tk.Frame(test_card, bg="#0d1117")
        out_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 6))

        self.inf_text_out = tk.Text(
            out_frame,
            bg="#0d1117",
            fg="#e6edf3",
            font=("Consolas", 9),
            relief=tk.FLAT,
            wrap=tk.WORD,
            height=5
        )
        self.inf_text_out.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 3. Speed Champions Recommendations Card
        champions_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        champions_card.pack(fill=tk.X, pady=(6, 0))

        champ_header = tk.Frame(champions_card, bg=self.COLOR_CARD)
        champ_header.pack(fill=tk.X, padx=15, pady=(6, 2))

        tk.Label(
            champ_header,
            text="⚡ ULTRA-FAST SPEED CHAMPIONS (Recommended Models for Instant Conversational Flow)",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_CYAN,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT)

        tk.Label(
            champions_card,
            text="💡 Heavy 4B–9B models are RAM bandwidth bound (~8–14 tok/s). For blazing fast 35 to 50+ tokens/second, pull these compact architectures:",
            font=("Segoe UI", 7),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=15, pady=(0, 4))

        chips_row = tk.Frame(champions_card, bg=self.COLOR_CARD)
        chips_row.pack(fill=tk.X, padx=15, pady=(0, 8))

        champions_list = [
            ("Llama 3.2 (1B)", "llama3.2:1b", "1.2 GB", "⚡ 35–50 tok/s", "Instant chat & reasoning"),
            ("DeepSeek R1 (1.5B)", "deepseek-r1:1.5b", "1.1 GB", "⚡ 30–45 tok/s", "Fast reasoning & logic"),
            ("Qwen 2.5 (1.5B)", "qwen2.5:1.5b", "1.0 GB", "⚡ 30–45 tok/s", "Top code & math benchmark"),
            ("Llama 3.2 (3B)", "llama3.2:3b", "2.0 GB", "⚡ 20–30 tok/s", "Intelligence sweetspot")
        ]

        for title, model_id, sz, spd, desc in champions_list:
            card = tk.Frame(chips_row, bg="#161b22", highlightbackground="#30363d", highlightthickness=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            top = tk.Frame(card, bg="#161b22")
            top.pack(fill=tk.X, padx=6, pady=(4, 1))
            tk.Label(top, text=title, font=("Segoe UI", 8, "bold"), fg=self.COLOR_TEXT, bg="#161b22").pack(side=tk.LEFT)
            tk.Label(top, text=spd, font=("Segoe UI", 7, "bold"), fg=self.COLOR_CYAN, bg="#161b22").pack(side=tk.RIGHT)

            tk.Label(card, text=f"{sz} • {desc}", font=("Segoe UI", 7), fg=self.COLOR_MUTED, bg="#161b22").pack(anchor="w", padx=6, pady=(0, 4))

            btn_qp = tk.Button(
                card,
                text="⚡ Quick Pull",
                font=("Segoe UI", 8, "bold"),
                bg="#1f3a4d",
                fg=self.COLOR_CYAN,
                activebackground=self.COLOR_CYAN,
                activeforeground="#000000",
                relief=tk.FLAT,
                pady=1,
                cursor="hand2",
                command=lambda m=model_id: self._quick_pull_model(m)
            )
            btn_qp.pack(fill=tk.X, padx=6, pady=(0, 4))

    def _start_inference_test(self):
        model = self.inf_model_cb.get().strip()
        if not model:
            messagebox.showwarning("Selection Required", "Please select a model from the dropdown to test.")
            return
        prompt = self.inf_prompt_var.get().strip()
        if not prompt:
            return

        ctx_val = 2048
        if hasattr(self, "inf_ctx_var"):
            try:
                ctx_val = int(self.inf_ctx_var.get())
            except Exception:
                pass

        self.btn_inf_run.config(state=tk.DISABLED)
        self.inf_text_out.delete("1.0", tk.END)
        self.inf_text_out.insert(tk.END, f"🚀 Executing boosted inference on {model} (context: {ctx_val})...\n\n")

        def run():
            try:
                stream = self.runtime_booster.generate_boosted_stream(model, prompt, num_ctx=ctx_val)
                for chunk in stream:
                    tok = chunk.get("token", "")
                    tps = chunk.get("current_tps", 0.0)
                    ttft = chunk.get("ttft")
                    toks = chunk.get("tokens_so_far", 0)
                    dur = chunk.get("total_duration_sec", 0.0)
                    pe_dur = chunk.get("prompt_eval_duration_sec")
                    pe_cnt = chunk.get("prompt_eval_count")

                    def update(t=tok, s=tps, tf=ttft, cnt=toks, d=dur, pc=pe_cnt, pd=pe_dur):
                        self.inf_text_out.insert(tk.END, t)
                        self.inf_text_out.see(tk.END)
                        self.lbl_inf_tps.config(text=f"{s:.1f} tok/s")
                        self.lbl_inf_tokens.config(text=f"{cnt} tokens")
                        self.lbl_inf_dur.config(text=f"Total Duration: {d:.2f}s")
                        if tf is not None:
                            pe_str = f"Prompt Eval: {(pc/pd):.1f} tok/s" if (pc and pd and pd > 0) else ""
                            self.lbl_inf_ttft.config(text=f"TTFT: {tf:.2f}s  |  {pe_str}")

                    self.after(0, update)
            except Exception as e:
                self.after(0, lambda err=str(e): self.inf_text_out.insert(tk.END, f"\n\n❌ Error connecting to Ollama: {err}\nEnsure Ollama is running."))
            finally:
                self.after(0, lambda: self.btn_inf_run.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _launch_inference_terminal(self):
        model = self.inf_model_cb.get().strip()
        if not model:
            messagebox.showwarning("Selection Required", "Please select a model from the dropdown.")
            return
        self.runtime_booster.launch_boosted_terminal(model)

    # ---------------- TAB 4: SPEED BENCHMARK ----------------

    def _build_tab_benchmark(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            card,
            text="DIAGNOSTIC ACCELERATION BENCHMARK",
            font=("Segoe UI", 12, "bold"),
            fg=self.COLOR_CYAN,
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=20, pady=(15, 6))

        tk.Label(
            card,
            text="Measures real download throughput by pulling identical model weights over single connection vs. 16x multi-stream.",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=20, pady=(0, 15))

        self.btn_run_bench = tk.Button(
            card,
            text="⚡ Run Live Speed Comparison",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_GREEN,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._start_benchmark
        )
        self.btn_run_bench.pack(anchor="w", padx=20, pady=(0, 15))

        self.bench_text = tk.Text(
            card,
            bg="#0d1117",
            fg=self.COLOR_TEXT,
            font=("Consolas", 10),
            relief=tk.FLAT,
            height=12
        )
        self.bench_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

    def _start_benchmark(self):
        self.btn_run_bench.config(state=tk.DISABLED)
        self.bench_text.delete("1.0", tk.END)
        self.bench_text.insert(tk.END, "Starting benchmark test against Ollama model weights...\n")

        def run():
            client = RegistryClient()
            try:
                manifest, _ = client.fetch_manifest("smollm:135m")
                layers = manifest.get("layers", [])
                m_layer = next((l for l in layers if "model" in l.get("mediaType", "")), layers[0])
                digest = m_layer["digest"]
                direct_url = client.resolve_blob_download_url("registry.ollama.ai", "library", "smollm", digest)

                test_bytes = 15 * 1024 * 1024  # 15 MB
                session = client.session

                self._bench_log("1. Measuring Single-Stream (Standard Ollama pull)...")
                t0 = time.time()
                r = session.get(direct_url, headers={"Range": f"bytes=0-{test_bytes-1}"}, stream=True, timeout=20)
                b1 = len(r.content)
                t1 = max(0.001, time.time() - t0)
                spd1 = (b1 / (1024 * 1024)) / t1
                self._bench_log(f"   ✓ Single connection: {spd1:.2f} MB/s in {t1:.2f}s")

                self._bench_log("2. Measuring Multi-Stream (16x Booster parallel streams)...")
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
                t2 = max(0.001, time.time() - t0)
                b2 = sum(chunks)
                spd2 = (b2 / (1024 * 1024)) / t2
                self._bench_log(f"   ✓ Multi-connection: {spd2:.2f} MB/s in {t2:.2f}s")

                boost = spd2 / spd1 if spd1 > 0 else 1.0
                self._bench_log("-" * 55)
                self._bench_log(f"⚡ RESULT: Booster is {boost:.1f}x FASTER!")
                self._bench_log(f"Large model pull (e.g. 5GB) estimated time:")
                self._bench_log(f"   Standard Ollama: ~{int(5120/spd1)} seconds")
                self._bench_log(f"   With Booster:    ~{int(5120/spd2)} seconds")
            except Exception as e:
                self._bench_log(f"❌ Benchmark error: {e}")
            finally:
                self.after(0, lambda: self.btn_run_bench.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _bench_log(self, text: str):
        self.after(0, lambda: [self.bench_text.insert(tk.END, text + "\n"), self.bench_text.see(tk.END)])

    # ---------------- TAB 4: SETTINGS & PROXY ----------------

    def _build_tab_settings(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Storage Path Card
        p_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        p_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(p_card, text="LOCAL OLLAMA MODELS DIRECTORY", font=("Segoe UI", 9, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(12, 4))
        tk.Label(p_card, text=self.manager.models_dir, font=("Consolas", 10), fg=self.COLOR_CYAN, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(0, 12))

        # Hugging Face Access Token Card
        hf_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        hf_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(hf_card, text="HUGGING FACE ACCESS TOKEN (FOR GATED MODELS)", font=("Segoe UI", 9, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(12, 4))
        tk.Label(
            hf_card,
            text="Required for gated Hugging Face models (e.g. LiquidAI/LFM, Meta-Llama). Get your free token at huggingface.co/settings/tokens",
            font=("Segoe UI", 8),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD
        ).pack(anchor="w", padx=15, pady=(0, 8))

        hf_row = tk.Frame(hf_card, bg=self.COLOR_CARD)
        hf_row.pack(fill=tk.X, padx=15, pady=(0, 12))

        existing_token = RegistryClient.get_hf_token() or ""
        self.hf_token_var = tk.StringVar(value=existing_token)

        self.hf_entry = tk.Entry(
            hf_row,
            textvariable=self.hf_token_var,
            font=("Consolas", 10),
            bg="#0d1117",
            fg=self.COLOR_TEXT,
            insertbackground=self.COLOR_CYAN,
            relief=tk.FLAT,
            bd=4,
            show="*" if existing_token else ""
        )
        self.hf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        btn_save_token = tk.Button(
            hf_row,
            text="Save Token",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_CARD,
            fg=self.COLOR_CYAN,
            relief=tk.FLAT,
            padx=12,
            pady=3,
            cursor="hand2",
            command=self._save_hf_token
        )
        btn_save_token.pack(side=tk.LEFT)

        # Local Proxy Card
        proxy_card = tk.Frame(container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        proxy_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(proxy_card, text="LOCAL ACCELERATOR PROXY SERVICE", font=("Segoe UI", 9, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(12, 4))
        tk.Label(proxy_card, text="Runs a local HTTP proxy on 127.0.0.1:11435 for CLI scripts and background services.", font=("Segoe UI", 9), fg=self.COLOR_TEXT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(0, 8))

        p_row = tk.Frame(proxy_card, bg=self.COLOR_CARD)
        p_row.pack(fill=tk.X, padx=15, pady=(0, 12))

        self.lbl_proxy_status = tk.Label(p_row, text="Status: Stopped", font=("Segoe UI", 9, "bold"), fg=self.COLOR_MUTED, bg=self.COLOR_CARD)
        self.lbl_proxy_status.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_toggle_proxy = tk.Button(
            p_row,
            text="Start Proxy Server",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_GREEN,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._toggle_proxy
        )
        self.btn_toggle_proxy.pack(side=tk.LEFT)

    def _save_hf_token(self):
        token = self.hf_token_var.get().strip()
        config_path = os.path.expanduser("~/.ollama/booster_config.json")
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            cfg = {}
            if os.path.isfile(config_path):
                import json
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    pass
            cfg["hf_token"] = token
            import json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self.hf_entry.config(show="*" if token else "")
            messagebox.showinfo("Token Saved", "Hugging Face Access Token saved successfully!\nGated model pulls will now use this token.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save token: {e}")

    def _toggle_proxy(self):
        if not self.proxy_server.is_running:
            if self.proxy_server.start():
                self.lbl_proxy_status.config(text="Status: Active on http://127.0.0.1:11435", fg=self.COLOR_GREEN_GLOW)
                self.btn_toggle_proxy.config(text="Stop Proxy Server", bg=self.COLOR_RED)
                messagebox.showinfo("Proxy Running", "Booster Proxy running on http://127.0.0.1:11435")
            else:
                messagebox.showerror("Error", "Port 11435 is already in use.")
        else:
            self.proxy_server.stop()
            self.lbl_proxy_status.config(text="Status: Stopped", fg=self.COLOR_MUTED)
            self.btn_toggle_proxy.config(text="Start Proxy Server", bg=self.COLOR_GREEN)

    # ---------------- STATUS BAR ----------------

    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg="#0a0d12", height=24)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        ver = self.manager.get_ollama_version()
        lbl_ollama = tk.Label(
            status_bar,
            text=f"Ollama: {ver}",
            font=("Segoe UI", 8),
            fg=self.COLOR_MUTED,
            bg="#0a0d12"
        )
        lbl_ollama.pack(side=tk.LEFT, padx=12, pady=3)

        lbl_version = tk.Label(
            status_bar,
            text="Ollama Download Booster v1.0.0",
            font=("Segoe UI", 8),
            fg=self.COLOR_MUTED,
            bg="#0a0d12"
        )
        lbl_version.pack(side=tk.RIGHT, padx=12, pady=3)


def start_gui():
    """Launch the Desktop Application."""
    app = BoosterApp()
    app.mainloop()
