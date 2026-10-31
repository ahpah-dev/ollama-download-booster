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
        self.title("Ollama Download Booster ⚡")
        self.geometry("940x720")
        self.minsize(840, 640)
        self.configure(bg=self.COLOR_BG)

        # Business Logic Components
        self.manager = OllamaManager()
        self.engine = BoosterEngine(ollama_manager=self.manager)
        self.proxy_server = BoosterProxyServer(port=11435, manager=self.manager)

        # State Variables
        self.is_boosted = tk.BooleanVar(value=True)  # Booster active by default
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
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 10))

        # Tab 1: Turbo Pull
        tab_pull = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_pull, text="  ⚡ Turbo Pull  ")
        self._build_tab_pull(tab_pull)

        # Tab 2: Installed Models
        tab_models = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_models, text="  📦 Models Manager  ")
        self._build_tab_models(tab_models)

        # Tab 3: Speed Benchmark
        tab_benchmark = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_benchmark, text="  📊 Speed Benchmark  ")
        self._build_tab_benchmark(tab_benchmark)

        # Tab 4: Booster Proxy & Settings
        tab_settings = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_settings, text="  ⚙️ Settings & Proxy  ")
        self._build_tab_settings(tab_settings)

        # Bottom Status Bar
        self._build_status_bar()

    def _build_header(self):
        """Top banner with app branding and big activation toggle."""
        header_frame = tk.Frame(self, bg=self.COLOR_BG)
        header_frame.pack(fill=tk.X, padx=20, pady=(15, 5))

        # Title & subtitle
        title_box = tk.Frame(header_frame, bg=self.COLOR_BG)
        title_box.pack(side=tk.LEFT)

        lbl_logo = tk.Label(
            title_box,
            text="⚡ OLLAMA DOWNLOAD BOOSTER",
            font=("Segoe UI", 16, "bold"),
            fg=self.COLOR_CYAN,
            bg=self.COLOR_BG
        )
        lbl_logo.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text="Multi-Stream Range Acceleration Engine for Ollama Models",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_BG
        )
        lbl_sub.pack(anchor="w")

        # Big Activation Card on Right
        self.btn_activate = tk.Button(
            header_frame,
            text="⚡ BOOSTER: ACTIVE (16x Streams)",
            font=("Segoe UI", 11, "bold"),
            bg="#1a472a",
            fg=self.COLOR_GREEN_GLOW,
            activebackground=self.COLOR_GREEN,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2",
            command=self._toggle_booster
        )
        self.btn_activate.pack(side=tk.RIGHT)

    def _toggle_booster(self):
        """Toggle booster state between Active and Standby."""
        new_state = not self.is_boosted.get()
        self.is_boosted.set(new_state)
        self._update_booster_ui()

    def _update_booster_ui(self):
        if self.is_boosted.get():
            threads = self.concurrency_var.get()
            self.btn_activate.config(
                text=f"⚡ BOOSTER: ACTIVE ({threads}x Streams)",
                bg="#1a472a",
                fg=self.COLOR_GREEN_GLOW
            )
            self._log_event(f"⚡ Booster Activated: {threads} parallel streams armed.")
        else:
            self.btn_activate.config(
                text="○ BOOSTER: STANDBY (1x Speed)",
                bg="#262c36",
                fg=self.COLOR_MUTED
            )
            self._log_event("Booster switched to Standby mode (single connection).")

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

        btn_run = tk.Button(
            bar,
            text="🚀 Run in Terminal",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_GREEN,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
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
        for m in models:
            size_str = f"{m['size_gb']} GB" if m['size_gb'] >= 1.0 else f"{m['size_mb']} MB"
            self.tree_models.insert("", tk.END, values=(m["name"], m["id"], size_str, m["tag"]))

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

    # ---------------- TAB 3: SPEED BENCHMARK ----------------

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
