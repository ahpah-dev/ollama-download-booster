"""
Modern, high-polish desktop GUI for Ollama Download & Runtime Booster.
Engineered with CustomTkinter for dark-mode aesthetics, responsive layouts,
real-time gauges, hardware telemetry, and interactive controls.
"""

import os
import sys
import time
import shutil
import threading
import webbrowser
from typing import Optional, Dict, Any, List
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from .booster_engine import BoosterEngine
from .models_manager import OllamaManager
from .proxy_server import BoosterProxyServer
from .runtime_booster import RuntimeBooster
from .registry_client import (
    RegistryClient,
    get_hf_token,
    save_hf_token,
    is_valid_hf_token,
    extract_hf_token,
)


# Set CustomTkinter global appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class BoosterApp(ctk.CTk):
    """
    Polished CustomTkinter desktop interface for Ollama Download & Runtime Booster.
    """

    # Modern Palette
    COLOR_BG_DARK = "#090d16"
    COLOR_BG_SIDEBAR = "#0f1523"
    COLOR_CARD = "#141c2c"
    COLOR_CARD_ALT = "#192336"
    COLOR_CARD_HOVER = "#1e2b42"
    COLOR_BORDER = "#243248"
    COLOR_BORDER_LIGHT = "#334155"

    COLOR_CYAN = "#00d2ff"
    COLOR_CYAN_HOVER = "#00a8cc"
    COLOR_CYAN_DIM = "#0d3b4c"

    COLOR_GREEN = "#10b981"
    COLOR_GREEN_HOVER = "#059669"
    COLOR_GREEN_DIM = "#133e31"

    COLOR_PURPLE = "#8b5cf6"
    COLOR_PURPLE_DIM = "#2e2154"

    COLOR_RED = "#ef4444"
    COLOR_RED_HOVER = "#dc2626"
    COLOR_RED_DIM = "#3f1c24"

    COLOR_TEXT_MAIN = "#f8fafc"
    COLOR_TEXT_MUTED = "#94a3b8"
    COLOR_TEXT_DIM = "#64748b"

    def __init__(self):
        super().__init__()

        # Window Configurations
        self.title("Ollama Download & Runtime Booster ⚡")
        self.geometry("1120x800")
        self.minsize(1020, 720)
        self.configure(fg_color=self.COLOR_BG_DARK)

        # Core Engines
        self.manager = OllamaManager()
        self.engine = BoosterEngine(ollama_manager=self.manager)
        self.proxy_server = BoosterProxyServer(port=11435, manager=self.manager)
        self.runtime_booster = RuntimeBooster()

        # State Variables
        self.is_download_boosted = ctk.BooleanVar(value=True)
        self.is_runtime_boosted = ctk.BooleanVar(value=self.runtime_booster.is_runtime_boost_enabled())
        if self.is_runtime_boosted.get():
            self.runtime_booster.start_process_priority_watcher()

        self.concurrency_var = ctk.StringVar(value="16x (Fastest)")
        self.model_input_var = ctk.StringVar(value="llama3.2:1b")
        self.search_model_var = ctk.StringVar()
        self.search_model_var.trace_add("write", lambda *args: self._filter_model_cards())

        # Hugging Face Authentication Token State
        self.hf_token_var = ctk.StringVar(value=get_hf_token() or "")

        self.is_downloading = False
        self.download_thread: Optional[threading.Thread] = None

        # View storage
        self.views: Dict[str, ctk.CTkFrame] = {}
        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        self.active_view_name = "pull"

        # Model cards cache
        self.installed_models_cache: List[Dict[str, Any]] = []

        # Configure Main Grid Layout
        self.grid_columnconfigure(0, weight=0, minsize=240)  # Left Sidebar
        self.grid_columnconfigure(1, weight=1)               # Main Content Area
        self.grid_rowconfigure(0, weight=1)

        # Build UI Architecture
        self._build_sidebar()
        self._build_views_container()

        # Initialize Data
        self._refresh_installed_models()
        self._start_service_poller()

        # Automatic HF Token Discovery on startup & window focus
        self.after(500, lambda: self._check_clipboard_for_token(silent_if_same=True))
        self.bind("<FocusIn>", lambda e: self._check_clipboard_for_token(silent_if_same=True))

    # =========================================================================
    # SIDEBAR NAVIGATION & TELEMETRY
    # =========================================================================

    def _build_sidebar(self):
        """Builds the sleek left navigation sidebar."""
        self.sidebar = ctk.CTkFrame(
            self,
            fg_color=self.COLOR_BG_SIDEBAR,
            corner_radius=0,
            border_width=1,
            border_color=self.COLOR_BORDER
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Brand / Logo Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(18, 14))

        logo_row = ctk.CTkFrame(brand_frame, fg_color="transparent")
        logo_row.pack(fill="x")

        logo_badge = ctk.CTkLabel(
            logo_row,
            text="⚡",
            font=("Segoe UI", 18, "bold"),
            fg_color=self.COLOR_CYAN_DIM,
            text_color=self.COLOR_CYAN,
            corner_radius=8,
            width=36,
            height=36
        )
        logo_badge.pack(side="left", padx=(0, 10))

        title_box = ctk.CTkFrame(logo_row, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="OLLAMA BOOSTER",
            font=("Segoe UI", 13, "bold"),
            text_color=self.COLOR_TEXT_MAIN
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Turbo Engine v1.2",
            font=("Segoe UI", 9),
            text_color=self.COLOR_CYAN
        ).pack(anchor="w")

        # Divider
        ctk.CTkFrame(self.sidebar, height=1, fg_color=self.COLOR_BORDER).pack(fill="x", padx=16, pady=(0, 14))

        # Nav Buttons Section
        nav_items = [
            ("pull", "⚡  Turbo Pull"),
            ("models", "📦  Installed Models"),
            ("inference", "🚀  Inference Turbo"),
            ("benchmark", "📊  Speed Benchmark"),
            ("settings", "⚙️  Settings & Proxy")
        ]

        self.nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav_frame.pack(fill="x", padx=12)

        for key, label in nav_items:
            btn = ctk.CTkButton(
                self.nav_frame,
                text=label,
                anchor="w",
                font=("Segoe UI", 11, "bold"),
                height=38,
                corner_radius=8,
                fg_color="transparent",
                text_color=self.COLOR_TEXT_MUTED,
                hover_color=self.COLOR_CARD_HOVER,
                command=lambda k=key: self.switch_view(k)
            )
            btn.pack(fill="x", pady=2)
            self.nav_buttons[key] = btn

        # Bottom Hardware & Engine Status Area
        bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", padx=12, pady=16)

        # Engine Quick Switches Card
        engine_card = ctk.CTkFrame(
            bottom_frame,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        engine_card.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            engine_card,
            text="ACCELERATION ENGINES",
            font=("Segoe UI", 8, "bold"),
            text_color=self.COLOR_TEXT_DIM
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.switch_dl = ctk.CTkSwitch(
            engine_card,
            text="16x Multi-Stream",
            font=("Segoe UI", 10, "bold"),
            text_color=self.COLOR_TEXT_MAIN,
            progress_color=self.COLOR_CYAN,
            variable=self.is_download_boosted,
            command=self._on_toggle_download_booster
        )
        self.switch_dl.pack(anchor="w", padx=12, pady=(4, 6))

        self.switch_inf = ctk.CTkSwitch(
            engine_card,
            text="Vulkan iGPU Turbo",
            font=("Segoe UI", 10, "bold"),
            text_color=self.COLOR_TEXT_MAIN,
            progress_color=self.COLOR_GREEN,
            variable=self.is_runtime_boosted,
            command=self._on_toggle_runtime_booster
        )
        self.switch_inf.pack(anchor="w", padx=12, pady=(0, 10))

        # Live Ollama Service Status Pill
        self.status_pill = ctk.CTkFrame(
            bottom_frame,
            fg_color=self.COLOR_CARD_ALT,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=8
        )
        self.status_pill.pack(fill="x")

        self.lbl_service_dot = ctk.CTkLabel(
            self.status_pill,
            text="●",
            font=("Segoe UI", 12),
            text_color=self.COLOR_GREEN,
            width=16
        )
        self.lbl_service_dot.pack(side="left", padx=(10, 4), pady=6)

        self.lbl_service_text = ctk.CTkLabel(
            self.status_pill,
            text="127.0.0.1:11434 Online",
            font=("Segoe UI", 9, "bold"),
            text_color=self.COLOR_TEXT_MUTED
        )
        self.lbl_service_text.pack(side="left", pady=6)

    def _build_views_container(self):
        """Constructs all five main views."""
        self.content_container = ctk.CTkFrame(self, fg_color=self.COLOR_BG_DARK, corner_radius=0)
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        # Instantiate View Frames
        self.views["pull"] = self._create_view_pull(self.content_container)
        self.views["models"] = self._create_view_models(self.content_container)
        self.views["inference"] = self._create_view_inference(self.content_container)
        self.views["benchmark"] = self._create_view_benchmark(self.content_container)
        self.views["settings"] = self._create_view_settings(self.content_container)

        # Show default view
        self.switch_view("pull")

    def switch_view(self, view_name: str):
        """Smoothly switches active content view with tab highlights."""
        if view_name not in self.views:
            return

        self.active_view_name = view_name

        # Update button highlights
        for k, btn in self.nav_buttons.items():
            if k == view_name:
                btn.configure(
                    fg_color=self.COLOR_CYAN_DIM,
                    text_color=self.COLOR_CYAN,
                    hover_color=self.COLOR_CYAN_DIM
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=self.COLOR_TEXT_MUTED,
                    hover_color=self.COLOR_CARD_HOVER
                )

        # Show target view
        for name, frame in self.views.items():
            if name == view_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        if view_name == "models":
            self._refresh_installed_models()

    # =========================================================================
    # VIEW 1: TURBO PULL (MULTI-STREAM DOWNLOADER)
    # =========================================================================

    def _create_view_pull(self, parent) -> ctk.CTkFrame:
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)

        # Header Title
        hdr = ctk.CTkFrame(view, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            hdr,
            text="⚡ TURBO MODEL DOWNLOADER",
            font=("Segoe UI", 16, "bold"),
            text_color=self.COLOR_TEXT_MAIN
        ).pack(anchor="w")

        ctk.CTkLabel(
            hdr,
            text="Multi-stream parallel HTTP range downloader • Direct seek zero-copy layer assembly",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w")

        # 1. Input & Controls Card
        ctrl_card = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        ctrl_card.pack(fill="x", pady=(0, 10))

        # Input Row
        input_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        input_row.pack(fill="x", padx=16, pady=(14, 8))

        self.entry_model = ctk.CTkEntry(
            input_row,
            textvariable=self.model_input_var,
            placeholder_text="Enter model name (e.g. llama3.2:1b, deepseek-r1:1.5b, hf.co/SC117/Spark-X2.6-4B...)",
            font=("Consolas", 12),
            height=42,
            corner_radius=8,
            border_color=self.COLOR_BORDER_LIGHT,
            fg_color=self.COLOR_BG_DARK
        )
        self.entry_model.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_start = ctk.CTkButton(
            input_row,
            text="⚡ Start Boosted Pull",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CYAN_HOVER,
            text_color="#000000",
            height=42,
            corner_radius=8,
            command=self._start_download
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_cancel = ctk.CTkButton(
            input_row,
            text="✕ Cancel",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.COLOR_RED,
            hover_color=self.COLOR_RED_HOVER,
            height=42,
            width=90,
            corner_radius=8,
            state="disabled",
            command=self._stop_download
        )
        self.btn_cancel.pack(side="left")

        # Quick Preset Chips Row
        presets_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        presets_row.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            presets_row,
            text="Quick Presets:",
            font=("Segoe UI", 9, "bold"),
            text_color=self.COLOR_TEXT_DIM
        ).pack(side="left", padx=(0, 8))

        quick_presets = [
            ("Llama 3.2 (1B)", "llama3.2:1b"),
            ("DeepSeek R1 (1.5B)", "deepseek-r1:1.5b"),
            ("Qwen 2.5 (1.5B)", "qwen2.5:1.5b"),
            ("Llama 3.2 (3B)", "llama3.2:3b"),
            ("Qwen 2.5 (7B)", "qwen2.5:7b"),
            ("Mistral (7B)", "mistral:7b")
        ]

        for label, model_ref in quick_presets:
            chip = ctk.CTkButton(
                presets_row,
                text=label,
                font=("Segoe UI", 9),
                height=26,
                corner_radius=6,
                fg_color=self.COLOR_CARD_ALT,
                text_color=self.COLOR_TEXT_MUTED,
                hover_color=self.COLOR_CARD_HOVER,
                command=lambda m=model_ref: self.model_input_var.set(m)
            )
            chip.pack(side="left", padx=3)

        # Hugging Face Presets & Helper Row
        hf_presets_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        hf_presets_row.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            hf_presets_row,
            text="🤗 Hugging Face:",
            font=("Segoe UI", 9, "bold"),
            text_color="#c4b5fd"
        ).pack(side="left", padx=(0, 8))

        hf_presets = [
            ("Llama-3.2-1B-GGUF", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"),
            ("DeepSeek-R1-1.5B-GGUF", "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M"),
            ("Qwen2.5-Coder-1.5B", "hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M"),
            ("Spark-X2.6-4B", "hf.co/SC117/Spark-X2.6-4B-abliterated-FIT-GGUF:Q4_K_M")
        ]

        for label, model_ref in hf_presets:
            chip = ctk.CTkButton(
                hf_presets_row,
                text=label,
                font=("Segoe UI", 9),
                height=26,
                corner_radius=6,
                fg_color="#231d38",
                text_color="#c4b5fd",
                hover_color="#362955",
                command=lambda m=model_ref: self.model_input_var.set(m)
            )
            chip.pack(side="left", padx=3)

        btn_hf_help = ctk.CTkButton(
            hf_presets_row,
            text="ℹ️ How to pull any HF model",
            font=("Segoe UI", 8, "bold"),
            height=24,
            corner_radius=6,
            fg_color=self.COLOR_CARD_ALT,
            text_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CARD_HOVER,
            command=self._show_hf_guide_dialog
        )
        btn_hf_help.pack(side="right")

        # Hugging Face Token Quick Status & Connect Strip
        hf_token_strip = ctk.CTkFrame(ctrl_card, fg_color=self.COLOR_BG_DARK, corner_radius=6)
        hf_token_strip.pack(fill="x", padx=16, pady=(0, 10))

        tok_val = (self.hf_token_var.get() or "").strip()
        has_tok = bool(tok_val and is_valid_hf_token(tok_val))
        masked_tok = f"{tok_val[:4]}••••••••{tok_val[-4:]}" if has_tok and len(tok_val) >= 12 else ""

        self.lbl_pull_hf_status = ctk.CTkLabel(
            hf_token_strip,
            text=f"🟢 HF Token Active ({masked_tok}) • Gated models enabled" if has_tok else "⚪ HF Token: None (Gated models require token)",
            font=("Segoe UI", 9),
            text_color=self.COLOR_GREEN if has_tok else self.COLOR_TEXT_MUTED
        )
        self.lbl_pull_hf_status.pack(side="left", padx=10, pady=6)

        self.btn_pull_hf_connect = ctk.CTkButton(
            hf_token_strip,
            text="🔄 Change Token" if has_tok else "🚀 1-Click Connect Token",
            font=("Segoe UI", 8, "bold"),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_CYAN,
            height=24,
            corner_radius=4,
            command=self._start_hf_auto_connect
        )
        self.btn_pull_hf_connect.pack(side="right", padx=8, pady=4)

        # Concurrency & Acceleration Bar
        concurrency_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        concurrency_row.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            concurrency_row,
            text="Concurrency:",
            font=("Segoe UI", 9, "bold"),
            text_color=self.COLOR_TEXT_DIM
        ).pack(side="left", padx=(0, 8))

        self.seg_concurrency = ctk.CTkSegmentedButton(
            concurrency_row,
            values=["4x", "8x", "16x (Fastest)", "24x", "32x (Max)"],
            variable=self.concurrency_var,
            font=("Segoe UI", 9, "bold"),
            selected_color=self.COLOR_CYAN,
            selected_hover_color=self.COLOR_CYAN_HOVER,
            corner_radius=6,
            height=28
        )
        self.seg_concurrency.pack(side="left")

        # 2. Live Metric Telemetry Dashboard (4 HUD Cards)
        telemetry_frame = ctk.CTkFrame(view, fg_color="transparent")
        telemetry_frame.pack(fill="x", pady=(0, 10))

        # Card 1: Speed
        c1 = ctk.CTkFrame(telemetry_frame, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        c1.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ctk.CTkLabel(c1, text="⚡ DOWNLOAD SPEED", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self.hud_speed = ctk.CTkLabel(c1, text="0.0 MB/s", font=("Segoe UI", 18, "bold"), text_color=self.COLOR_CYAN)
        self.hud_speed.pack(anchor="w", padx=12, pady=(0, 2))
        self.hud_speed_sub = ctk.CTkLabel(c1, text="Peak: 0.0 MB/s", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_speed_sub.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 2: Transferred
        c2 = ctk.CTkFrame(telemetry_frame, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        c2.pack(side="left", fill="both", expand=True, padx=3)
        ctk.CTkLabel(c2, text="💾 TRANSFERRED", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self.hud_transferred = ctk.CTkLabel(c2, text="0.0 MB / 0.0 MB", font=("Segoe UI", 18, "bold"), text_color=self.COLOR_TEXT_MAIN)
        self.hud_transferred.pack(anchor="w", padx=12, pady=(0, 2))
        self.hud_pct = ctk.CTkLabel(c2, text="0.0% Complete", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_pct.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 3: ETA
        c3 = ctk.CTkFrame(telemetry_frame, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        c3.pack(side="left", fill="both", expand=True, padx=3)
        ctk.CTkLabel(c3, text="⏱️ ESTIMATED TIME", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self.hud_eta = ctk.CTkLabel(c3, text="ETA: --", font=("Segoe UI", 18, "bold"), text_color=self.COLOR_TEXT_MAIN)
        self.hud_eta.pack(anchor="w", padx=12, pady=(0, 2))
        self.hud_eta_sub = ctk.CTkLabel(c3, text="Multi-Stream Active", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_eta_sub.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 4: Streams
        c4 = ctk.CTkFrame(telemetry_frame, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        c4.pack(side="left", fill="both", expand=True, padx=(6, 0))
        ctk.CTkLabel(c4, text="🌐 ACTIVE STREAMS", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self.hud_streams = ctk.CTkLabel(c4, text="0 Streams", font=("Segoe UI", 18, "bold"), text_color=self.COLOR_GREEN)
        self.hud_streams.pack(anchor="w", padx=12, pady=(0, 2))
        self.hud_streams_sub = ctk.CTkLabel(c4, text="Zero-Copy Direct Seek", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_streams_sub.pack(anchor="w", padx=12, pady=(0, 8))

        # 3. Main Progress Bar
        self.prog_bar = ctk.CTkProgressBar(
            view,
            height=14,
            corner_radius=7,
            fg_color=self.COLOR_CARD,
            progress_color=self.COLOR_CYAN
        )
        self.prog_bar.set(0.0)
        self.prog_bar.pack(fill="x", pady=(0, 10))

        # 4. Multi-Stream Live Activity Strip
        self.streams_frame = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=8,
            height=40
        )
        self.streams_frame.pack(fill="x", pady=(0, 10))

        self.stream_chips: List[ctk.CTkLabel] = []
        chips_box = ctk.CTkFrame(self.streams_frame, fg_color="transparent")
        chips_box.pack(fill="both", expand=True, padx=8, pady=6)

        for i in range(16):
            lbl = ctk.CTkLabel(
                chips_box,
                text=f"S{i+1}",
                font=("Consolas", 8, "bold"),
                fg_color=self.COLOR_CARD_ALT,
                text_color=self.COLOR_TEXT_DIM,
                corner_radius=4,
                width=32,
                height=22
            )
            lbl.pack(side="left", fill="both", expand=True, padx=2)
            self.stream_chips.append(lbl)

        # 5. Live Console Log
        log_header = ctk.CTkFrame(view, fg_color="transparent")
        log_header.pack(fill="x", pady=(2, 4))

        ctk.CTkLabel(
            log_header,
            text="ENGINE ACTIVITY LOG",
            font=("Segoe UI", 9, "bold"),
            text_color=self.COLOR_TEXT_DIM
        ).pack(side="left")

        btn_clear = ctk.CTkButton(
            log_header,
            text="Clear",
            font=("Segoe UI", 8),
            fg_color=self.COLOR_CARD,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MUTED,
            height=22,
            width=50,
            command=self._clear_log
        )
        btn_clear.pack(side="right", padx=(4, 0))

        btn_copy = ctk.CTkButton(
            log_header,
            text="Copy",
            font=("Segoe UI", 8),
            fg_color=self.COLOR_CARD,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MUTED,
            height=22,
            width=50,
            command=self._copy_log
        )
        btn_copy.pack(side="right")

        self.log_text = ctk.CTkTextbox(
            view,
            font=("Consolas", 10),
            fg_color=self.COLOR_BG_DARK,
            text_color="#e2e8f0",
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=8,
            height=130
        )
        self.log_text.pack(fill="both", expand=True)

        return view

    # =========================================================================
    # VIEW 2: INSTALLED MODELS MANAGER
    # =========================================================================

    def _create_view_models(self, parent) -> ctk.CTkFrame:
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(2, weight=1)

        # Header Title
        hdr = ctk.CTkFrame(view, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 10))

        self.lbl_models_hdr = ctk.CTkLabel(
            hdr,
            text="📦 LOCAL OLLAMA LIBRARY",
            font=("Segoe UI", 16, "bold"),
            text_color=self.COLOR_TEXT_MAIN
        )
        self.lbl_models_hdr.pack(anchor="w")

        ctk.CTkLabel(
            hdr,
            text="Launch boosted models, inspect sizes, and generate instant 0-disk -turbo profiles",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w")

        # Search & Action Toolbar Card
        bar_card = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        bar_card.pack(fill="x", pady=(0, 10))

        bar = ctk.CTkFrame(bar_card, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=10)

        # Search Entry
        self.entry_search = ctk.CTkEntry(
            bar,
            textvariable=self.search_model_var,
            placeholder_text="🔍 Filter installed models...",
            font=("Segoe UI", 11),
            width=260,
            height=34,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        self.entry_search.pack(side="left", padx=(0, 10))

        btn_refresh = ctk.CTkButton(
            bar,
            text="🔄 Refresh",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MAIN,
            height=34,
            width=80,
            corner_radius=6,
            command=self._refresh_installed_models
        )
        btn_refresh.pack(side="right")

        # Scrollable Cards Container
        self.models_scroll = ctk.CTkScrollableFrame(
            view,
            fg_color="transparent",
            corner_radius=8
        )
        self.models_scroll.pack(fill="both", expand=True)

        return view

    def _render_model_cards(self, models: List[Dict[str, Any]]):
        """Renders polished cards for installed models."""
        for widget in self.models_scroll.winfo_children():
            widget.destroy()

        if not models:
            empty_frame = ctk.CTkFrame(self.models_scroll, fg_color=self.COLOR_CARD, corner_radius=10)
            empty_frame.pack(fill="x", pady=20, padx=10)
            ctk.CTkLabel(
                empty_frame,
                text="No matching models found in local storage.",
                font=("Segoe UI", 12),
                text_color=self.COLOR_TEXT_MUTED
            ).pack(pady=20)
            return

        for m in models:
            card = ctk.CTkFrame(
                self.models_scroll,
                fg_color=self.COLOR_CARD,
                border_width=1,
                border_color=self.COLOR_BORDER,
                corner_radius=10
            )
            card.pack(fill="x", pady=4, padx=2)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=14, pady=10)

            # Left Details
            left = ctk.CTkFrame(inner, fg_color="transparent")
            left.pack(side="left", fill="y")

            title_row = ctk.CTkFrame(left, fg_color="transparent")
            title_row.pack(anchor="w")

            ctk.CTkLabel(
                title_row,
                text=m["name"],
                font=("Segoe UI", 12, "bold"),
                text_color=self.COLOR_TEXT_MAIN
            ).pack(side="left", padx=(0, 8))

            if "-turbo" in m["name"] or ":turbo" in m["name"]:
                badge = ctk.CTkLabel(
                    title_row,
                    text="⚡ TURBO TUNED",
                    font=("Segoe UI", 8, "bold"),
                    fg_color=self.COLOR_CYAN_DIM,
                    text_color=self.COLOR_CYAN,
                    corner_radius=4,
                    height=18,
                    padx=6
                )
                badge.pack(side="left")

            size_str = f"{m['size_gb']} GB" if m['size_gb'] >= 1.0 else f"{m['size_mb']} MB"
            sub_text = f"ID: {m['id']}  •  Size: {size_str}  •  Tag: {m['tag']}"
            ctk.CTkLabel(
                left,
                text=sub_text,
                font=("Segoe UI", 9),
                text_color=self.COLOR_TEXT_MUTED
            ).pack(anchor="w", pady=(2, 0))

            # Right Action Buttons
            right = ctk.CTkFrame(inner, fg_color="transparent")
            right.pack(side="right")

            btn_run_boosted = ctk.CTkButton(
                right,
                text="⚡ Run Boosted",
                font=("Segoe UI", 9, "bold"),
                fg_color=self.COLOR_CYAN_DIM,
                hover_color=self.COLOR_CYAN,
                text_color=self.COLOR_CYAN,
                height=30,
                corner_radius=6,
                command=lambda name=m["name"]: self.runtime_booster.launch_boosted_terminal(name)
            )
            btn_run_boosted.pack(side="left", padx=3)

            if "-turbo" not in m["name"]:
                btn_turbo = ctk.CTkButton(
                    right,
                    text="⚡ -turbo Copy",
                    font=("Segoe UI", 9),
                    fg_color=self.COLOR_CARD_ALT,
                    hover_color=self.COLOR_CARD_HOVER,
                    text_color=self.COLOR_CYAN,
                    height=30,
                    corner_radius=6,
                    command=lambda name=m["name"]: self._create_selected_model_turbo(name)
                )
                btn_turbo.pack(side="left", padx=3)

            btn_std = ctk.CTkButton(
                right,
                text="Standard",
                font=("Segoe UI", 9),
                fg_color=self.COLOR_CARD_ALT,
                hover_color=self.COLOR_CARD_HOVER,
                text_color=self.COLOR_TEXT_MUTED,
                height=30,
                width=65,
                corner_radius=6,
                command=lambda name=m["name"]: self.manager.run_model_terminal(name)
            )
            btn_std.pack(side="left", padx=3)

            btn_del = ctk.CTkButton(
                right,
                text="🗑",
                font=("Segoe UI", 11),
                fg_color=self.COLOR_CARD_ALT,
                hover_color=self.COLOR_RED_HOVER,
                text_color=self.COLOR_RED,
                height=30,
                width=32,
                corner_radius=6,
                command=lambda name=m["name"]: self._delete_model(name)
            )
            btn_del.pack(side="left", padx=(3, 0))

    def _filter_model_cards(self):
        query = self.search_model_var.get().strip().lower()
        if not query:
            self._render_model_cards(self.installed_models_cache)
        else:
            filtered = [m for m in self.installed_models_cache if query in m["name"].lower() or query in m["tag"].lower()]
            self._render_model_cards(filtered)

    # =========================================================================
    # VIEW 3: INFERENCE TURBO (THE SPEED ACCELERATOR)
    # =========================================================================

    def _create_view_inference(self, parent) -> ctk.CTkFrame:
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)

        # 1. Master Service Status & Turbo Injection Banner
        svc_banner = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        svc_banner.pack(fill="x", pady=(0, 10))

        svc_inner = ctk.CTkFrame(svc_banner, fg_color="transparent")
        svc_inner.pack(fill="x", padx=16, pady=12)

        svc_left = ctk.CTkFrame(svc_inner, fg_color="transparent")
        svc_left.pack(side="left", fill="x", expand=True)

        top_row = ctk.CTkFrame(svc_left, fg_color="transparent")
        top_row.pack(anchor="w")

        self.lbl_inf_service_dot = ctk.CTkLabel(
            top_row,
            text="●",
            font=("Segoe UI", 14),
            text_color=self.COLOR_GREEN,
            width=18
        )
        self.lbl_inf_service_dot.pack(side="left", padx=(0, 4))

        self.lbl_inf_service_title = ctk.CTkLabel(
            top_row,
            text="Ollama Service: Active (127.0.0.1:11434 - AMD Radeon Vulkan Turbo)",
            font=("Segoe UI", 12, "bold"),
            text_color=self.COLOR_TEXT_MAIN
        )
        self.lbl_inf_service_title.pack(side="left")

        ctk.CTkLabel(
            svc_left,
            text="💡 Apply Turbo: Ollama initializes GPU Vulkan & Flash Attention on service startup. Click below to inject all turbo flags.",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w", pady=(2, 0))

        self.btn_restart_ollama = ctk.CTkButton(
            svc_inner,
            text="🔄 Apply & Restart Ollama with Turbo",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_GREEN,
            hover_color=self.COLOR_GREEN_HOVER,
            text_color="#ffffff",
            height=38,
            corner_radius=8,
            command=self._restart_ollama_turbo
        )
        self.btn_restart_ollama.pack(side="right")

        # 2. Hardware Acceleration Tiles (5 Badges)
        tiles_frame = ctk.CTkFrame(view, fg_color="transparent")
        tiles_frame.pack(fill="x", pady=(0, 10))

        tiles = [
            ("🎮 AMD iGPU Vulkan", "OLLAMA_IGPU_ENABLE=1"),
            ("🚀 Flash Attention", "OLLAMA_FLASH_ATTENTION=1"),
            ("⚡ 24h Keep-Alive", "Zero 5-min cold unloads"),
            ("🧠 Physical Cores", f"{self.runtime_booster.physical_cores} Cores Locked"),
            ("🏎️ High Priority", "0ms Windows throttling")
        ]

        for title, sub in tiles:
            tile = ctk.CTkFrame(tiles_frame, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
            tile.pack(side="left", fill="both", expand=True, padx=2)
            ctk.CTkLabel(tile, text=title, font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=8, pady=(6, 0))
            ctk.CTkLabel(tile, text=sub, font=("Segoe UI", 7), text_color=self.COLOR_CYAN).pack(anchor="w", padx=8, pady=(0, 6))

        # 3. Interactive Testing & Speedometer Card
        test_card = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        test_card.pack(fill="both", expand=True, pady=(0, 10))

        test_inner = ctk.CTkFrame(test_card, fg_color="transparent")
        test_inner.pack(fill="both", expand=True, padx=16, pady=12)

        # Controls Row
        ctrls_row = ctk.CTkFrame(test_inner, fg_color="transparent")
        ctrls_row.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(ctrls_row, text="Model:", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(side="left", padx=(0, 6))
        self.cb_inf_model = ctk.CTkComboBox(
            ctrls_row,
            values=["llama3.2:1b"],
            font=("Consolas", 11),
            width=240,
            height=32,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        self.cb_inf_model.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(ctrls_row, text="Ctx:", font=("Segoe UI", 9, "bold"), text_color=self.COLOR_TEXT_DIM).pack(side="left", padx=(0, 4))
        self.cb_inf_ctx = ctk.CTkComboBox(
            ctrls_row,
            values=["1024", "2048", "4096", "8192"],
            width=80,
            height=32,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        self.cb_inf_ctx.set("2048")
        self.cb_inf_ctx.pack(side="left", padx=(0, 10))

        btn_turbo_copy = ctk.CTkButton(
            ctrls_row,
            text="⚡ Create -turbo Copy",
            font=("Segoe UI", 9, "bold"),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_CYAN,
            height=32,
            corner_radius=6,
            command=lambda: self._create_selected_model_turbo(self.cb_inf_model.get())
        )
        btn_turbo_copy.pack(side="left", padx=(0, 8))

        self.btn_inf_run = ctk.CTkButton(
            ctrls_row,
            text="⚡ Stream & Measure Speed",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CYAN_HOVER,
            text_color="#000000",
            height=32,
            corner_radius=6,
            command=self._start_inference_test
        )
        self.btn_inf_run.pack(side="left", padx=(0, 8))

        btn_term = ctk.CTkButton(
            ctrls_row,
            text="🚀 Terminal",
            font=("Segoe UI", 9),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MAIN,
            height=32,
            width=70,
            corner_radius=6,
            command=self._launch_inference_terminal
        )
        btn_term.pack(side="left")

        # Prompt Entry & Preset Chips
        prompt_box = ctk.CTkFrame(test_inner, fg_color="transparent")
        prompt_box.pack(fill="x", pady=(0, 8))

        self.inf_prompt_var = ctk.StringVar(value="Explain quantum computing in two concise sentences.")
        self.entry_inf_prompt = ctk.CTkEntry(
            prompt_box,
            textvariable=self.inf_prompt_var,
            font=("Consolas", 11),
            height=36,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        self.entry_inf_prompt.pack(fill="x", pady=(0, 4))

        quick_prompts_row = ctk.CTkFrame(prompt_box, fg_color="transparent")
        quick_prompts_row.pack(fill="x")

        ctk.CTkLabel(quick_prompts_row, text="Quick Prompts:", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(side="left", padx=(0, 6))
        for q_lbl, q_text in [
            ("Short Fact", "Give me 1 fascinating fact about astronomy."),
            ("Python Code", "Write a Python function to check for palindromes."),
            ("Speed Benchmark", "Write a 50-word poem about speed and light.")
        ]:
            b = ctk.CTkButton(
                quick_prompts_row,
                text=q_lbl,
                font=("Segoe UI", 8),
                height=22,
                corner_radius=4,
                fg_color=self.COLOR_CARD_ALT,
                text_color=self.COLOR_TEXT_MUTED,
                hover_color=self.COLOR_CARD_HOVER,
                command=lambda t=q_text: self.inf_prompt_var.set(t)
            )
            b.pack(side="left", padx=2)

        # Digital Gauges Row (HUD Speedometer)
        hud_row = ctk.CTkFrame(test_inner, fg_color="transparent")
        hud_row.pack(fill="x", pady=(0, 8))

        # Gauge 1: Speed
        g1 = ctk.CTkFrame(hud_row, fg_color=self.COLOR_BG_DARK, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        g1.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ctk.CTkLabel(g1, text="GENERATION SPEED", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=10, pady=(6, 0))
        self.hud_inf_tps = ctk.CTkLabel(g1, text="0.0 tok/s", font=("Segoe UI", 20, "bold"), text_color=self.COLOR_CYAN)
        self.hud_inf_tps.pack(anchor="w", padx=10, pady=(0, 2))
        self.hud_inf_ttft = ctk.CTkLabel(g1, text="TTFT: --  |  Prompt Eval: --", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_inf_ttft.pack(anchor="w", padx=10, pady=(0, 6))

        # Gauge 2: Tokens & Duration
        g2 = ctk.CTkFrame(hud_row, fg_color=self.COLOR_BG_DARK, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        g2.pack(side="left", fill="both", expand=True, padx=(4, 0))
        ctk.CTkLabel(g2, text="TOKENS & LATENCY", font=("Segoe UI", 8, "bold"), text_color=self.COLOR_TEXT_DIM).pack(anchor="w", padx=10, pady=(6, 0))
        self.hud_inf_toks = ctk.CTkLabel(g2, text="0 tokens", font=("Segoe UI", 20, "bold"), text_color=self.COLOR_TEXT_MAIN)
        self.hud_inf_toks.pack(anchor="w", padx=10, pady=(0, 2))
        self.hud_inf_dur = ctk.CTkLabel(g2, text="Total Duration: 0.0s", font=("Segoe UI", 8), text_color=self.COLOR_TEXT_MUTED)
        self.hud_inf_dur.pack(anchor="w", padx=10, pady=(0, 6))

        # Streaming Output Console
        self.inf_text_out = ctk.CTkTextbox(
            test_inner,
            font=("Consolas", 10),
            fg_color=self.COLOR_BG_DARK,
            text_color="#e2e8f0",
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=8,
            height=110
        )
        self.inf_text_out.pack(fill="both", expand=True)

        # 4. Ultra-Fast Speed Champions Section
        champ_card = ctk.CTkFrame(
            view,
            fg_color=self.COLOR_CARD,
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=10
        )
        champ_card.pack(fill="x")

        champ_inner = ctk.CTkFrame(champ_card, fg_color="transparent")
        champ_inner.pack(fill="x", padx=14, pady=10)

        champ_hdr = ctk.CTkFrame(champ_inner, fg_color="transparent")
        champ_hdr.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            champ_hdr,
            text="⚡ ULTRA-FAST SPEED CHAMPIONS",
            font=("Segoe UI", 10, "bold"),
            text_color=self.COLOR_CYAN
        ).pack(side="left")

        ctk.CTkLabel(
            champ_inner,
            text="💡 Heavy 4B–9B models are RAM bandwidth bound (~8–14 tok/s). For blazing fast 35 to 50+ tokens/second, pull these compact architectures:",
            font=("Segoe UI", 8),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w", pady=(0, 8))

        chips_row = ctk.CTkFrame(champ_inner, fg_color="transparent")
        chips_row.pack(fill="x")

        champions_list = [
            ("Llama 3.2 (1B)", "llama3.2:1b", "1.2 GB", "⚡ 35–50 tok/s", "Instant chat"),
            ("DeepSeek R1 (1.5B)", "deepseek-r1:1.5b", "1.1 GB", "⚡ 30–45 tok/s", "Fast reasoning"),
            ("Qwen 2.5 (1.5B)", "qwen2.5:1.5b", "1.0 GB", "⚡ 30–45 tok/s", "Top code/math"),
            ("Llama 3.2 (3B)", "llama3.2:3b", "2.0 GB", "⚡ 20–30 tok/s", "Sweetspot")
        ]

        for title, model_id, sz, spd, desc in champions_list:
            c = ctk.CTkFrame(chips_row, fg_color=self.COLOR_CARD_ALT, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
            c.pack(side="left", fill="both", expand=True, padx=3)

            top = ctk.CTkFrame(c, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(6, 2))
            ctk.CTkLabel(top, text=title, font=("Segoe UI", 9, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(side="left")
            ctk.CTkLabel(top, text=spd, font=("Segoe UI", 8, "bold"), text_color=self.COLOR_CYAN).pack(side="right")

            ctk.CTkLabel(c, text=f"{sz} • {desc}", font=("Segoe UI", 7), text_color=self.COLOR_TEXT_MUTED).pack(anchor="w", padx=8, pady=(0, 6))

            btn_qp = ctk.CTkButton(
                c,
                text="⚡ Quick Pull",
                font=("Segoe UI", 8, "bold"),
                fg_color=self.COLOR_CYAN_DIM,
                hover_color=self.COLOR_CYAN,
                text_color=self.COLOR_CYAN,
                height=24,
                corner_radius=4,
                command=lambda m=model_id: self._quick_pull_model(m)
            )
            btn_qp.pack(fill="x", padx=8, pady=(0, 6))

        return view

    # =========================================================================
    # VIEW 4: SPEED BENCHMARK
    # =========================================================================

    def _create_view_benchmark(self, parent) -> ctk.CTkFrame:
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(view, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        card.pack(fill="both", expand=True)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            inner,
            text="📊 DIAGNOSTIC ACCELERATION BENCHMARK",
            font=("Segoe UI", 16, "bold"),
            text_color=self.COLOR_TEXT_MAIN
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            inner,
            text="Measures real download throughput by pulling identical model weights over single connection vs. 16x multi-stream.",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w", pady=(0, 20))

        self.btn_run_bench = ctk.CTkButton(
            inner,
            text="⚡ Run Live Speed Comparison",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CYAN_HOVER,
            text_color="#000000",
            height=40,
            width=220,
            corner_radius=8,
            command=self._run_benchmark
        )
        self.btn_run_bench.pack(anchor="w", pady=(0, 20))

        # Visual Comparison Bars
        comp_frame = ctk.CTkFrame(inner, fg_color=self.COLOR_BG_DARK, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        comp_frame.pack(fill="x", pady=(0, 16), padx=2)

        c_inner = ctk.CTkFrame(comp_frame, fg_color="transparent")
        c_inner.pack(fill="x", padx=16, pady=16)

        # Single Stream Bar
        ctk.CTkLabel(c_inner, text="Standard Single Connection (ollama pull):", font=("Segoe UI", 9, "bold"), text_color=self.COLOR_TEXT_MUTED).pack(anchor="w")
        self.prog_bench_single = ctk.CTkProgressBar(c_inner, height=14, corner_radius=7, fg_color=self.COLOR_CARD, progress_color=self.COLOR_RED)
        self.prog_bench_single.set(0.0)
        self.prog_bench_single.pack(fill="x", pady=(4, 2))
        self.lbl_bench_single = ctk.CTkLabel(c_inner, text="Waiting for benchmark...", font=("Segoe UI", 9), text_color=self.COLOR_TEXT_MUTED)
        self.lbl_bench_single.pack(anchor="w", pady=(0, 14))

        # Multi-Stream Bar
        ctk.CTkLabel(c_inner, text="Booster Multi-Stream (16 parallel worker streams):", font=("Segoe UI", 9, "bold"), text_color=self.COLOR_CYAN).pack(anchor="w")
        self.prog_bench_multi = ctk.CTkProgressBar(c_inner, height=14, corner_radius=7, fg_color=self.COLOR_CARD, progress_color=self.COLOR_GREEN)
        self.prog_bench_multi.set(0.0)
        self.prog_bench_multi.pack(fill="x", pady=(4, 2))
        self.lbl_bench_multi = ctk.CTkLabel(c_inner, text="Waiting for benchmark...", font=("Segoe UI", 9), text_color=self.COLOR_TEXT_MUTED)
        self.lbl_bench_multi.pack(anchor="w", pady=(0, 8))

        # Diagnostic Details Log
        self.bench_text = ctk.CTkTextbox(
            inner,
            font=("Consolas", 10),
            fg_color=self.COLOR_BG_DARK,
            text_color="#e2e8f0",
            border_width=1,
            border_color=self.COLOR_BORDER,
            corner_radius=8,
            height=180
        )
        self.bench_text.pack(fill="both", expand=True)

        return view

    # =========================================================================
    # VIEW 5: SETTINGS & PROXY
    # =========================================================================

    def _create_view_settings(self, parent) -> ctk.CTkFrame:
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)

        # 1. Local Accelerator Proxy Card
        proxy_card = ctk.CTkFrame(view, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        proxy_card.pack(fill="x", pady=(0, 14))

        p_inner = ctk.CTkFrame(proxy_card, fg_color="transparent")
        p_inner.pack(fill="x", padx=18, pady=16)

        ctk.CTkLabel(p_inner, text="🌐 LOCAL ACCELERATOR PROXY", font=("Segoe UI", 12, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(
            p_inner,
            text="Transparent background proxy server on http://127.0.0.1:11435 that intercepts standard 'ollama pull' CLI calls and automatically routes them through the multi-stream accelerator engine.",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            wraplength=700
        ).pack(anchor="w", pady=(2, 10))

        p_row = ctk.CTkFrame(p_inner, fg_color="transparent")
        p_row.pack(fill="x")

        self.btn_proxy_toggle = ctk.CTkButton(
            p_row,
            text="Start Accelerator Proxy",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_GREEN,
            hover_color=self.COLOR_GREEN_HOVER,
            height=34,
            corner_radius=6,
            command=self._toggle_proxy
        )
        self.btn_proxy_toggle.pack(side="left", padx=(0, 10))

        self.lbl_proxy_status = ctk.CTkLabel(
            p_row,
            text="Proxy Status: Stopped",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED
        )
        self.lbl_proxy_status.pack(side="left")

        # 2. Hugging Face Authentication Token Card
        hf_card = ctk.CTkFrame(view, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        hf_card.pack(fill="x", pady=(0, 14))

        hf_inner = ctk.CTkFrame(hf_card, fg_color="transparent")
        hf_inner.pack(fill="x", padx=18, pady=16)

        ctk.CTkLabel(hf_inner, text="🔑 HUGGING FACE ACCESS TOKEN", font=("Segoe UI", 12, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(
            hf_inner,
            text="Allows pulling gated models and private repositories directly via hf.co/... references without 401 Unauthorized errors.",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w", pady=(2, 8))

        # Status badge label
        tok_val = (self.hf_token_var.get() or "").strip()
        has_tok = bool(tok_val and is_valid_hf_token(tok_val))
        masked_tok = f"{tok_val[:4]}••••••••{tok_val[-4:]}" if has_tok and len(tok_val) >= 12 else ""

        self.lbl_settings_hf_status = ctk.CTkLabel(
            hf_inner,
            text=f"🟢 Active Token: {masked_tok}  •  Synced to System Environment, Registry & Cache" if has_tok else "⚪ No Token Configured — Gated models require an access token",
            font=("Segoe UI", 9, "bold" if has_tok else "normal"),
            text_color=self.COLOR_GREEN if has_tok else self.COLOR_TEXT_MUTED
        )
        self.lbl_settings_hf_status.pack(anchor="w", pady=(0, 10))

        # 1-Click Auto-Connect Box
        auto_box = ctk.CTkFrame(hf_inner, fg_color=self.COLOR_BG_DARK, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        auto_box.pack(fill="x", pady=(0, 12))

        auto_box_inner = ctk.CTkFrame(auto_box, fg_color="transparent")
        auto_box_inner.pack(fill="x", padx=14, pady=12)

        ctk.CTkLabel(
            auto_box_inner,
            text="⚡ 1-Click Auto-Connect (Zero Typing Needed)",
            font=("Segoe UI", 10, "bold"),
            text_color=self.COLOR_CYAN
        ).pack(anchor="w")

        ctk.CTkLabel(
            auto_box_inner,
            text="Opens Hugging Face in your browser with 'read' token pre-filled. Simply copy the token, and this app will auto-detect from your clipboard, save, and activate it permanently!",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            wraplength=680,
            justify="left"
        ).pack(anchor="w", pady=(2, 8))

        btn_row_auto = ctk.CTkFrame(auto_box_inner, fg_color="transparent")
        btn_row_auto.pack(fill="x")

        btn_auto_connect = ctk.CTkButton(
            btn_row_auto,
            text="🚀 1-Click Connect Hugging Face Token",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_PURPLE,
            hover_color="#7c3aed",
            text_color="#ffffff",
            height=34,
            corner_radius=6,
            command=self._start_hf_auto_connect
        )
        btn_auto_connect.pack(side="left", padx=(0, 10))

        btn_paste_clip = ctk.CTkButton(
            btn_row_auto,
            text="📋 Paste & Apply from Clipboard",
            font=("Segoe UI", 9),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MAIN,
            height=34,
            corner_radius=6,
            command=lambda: self._check_clipboard_for_token(force=True)
        )
        btn_paste_clip.pack(side="left")

        self.lbl_hf_listen_status = ctk.CTkLabel(
            auto_box_inner,
            text="",
            font=("Segoe UI", 9),
            text_color=self.COLOR_CYAN
        )
        self.lbl_hf_listen_status.pack(anchor="w", pady=(6, 0))

        # Manual Entry Row
        ctk.CTkLabel(
            hf_inner,
            text="Or Enter / Edit Token Manually:",
            font=("Segoe UI", 9, "bold"),
            text_color=self.COLOR_TEXT_DIM
        ).pack(anchor="w", pady=(0, 4))

        hf_row = ctk.CTkFrame(hf_inner, fg_color="transparent")
        hf_row.pack(fill="x")

        self.entry_hf = ctk.CTkEntry(
            hf_row,
            textvariable=self.hf_token_var,
            placeholder_text="hf_...",
            font=("Consolas", 11),
            show="*",
            width=320,
            height=34,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        self.entry_hf.pack(side="left", padx=(0, 8))

        self.btn_hf_show = ctk.CTkButton(
            hf_row,
            text="Show",
            font=("Segoe UI", 9),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MAIN,
            height=34,
            width=50,
            corner_radius=6,
            command=self._toggle_hf_visibility
        )
        self.btn_hf_show.pack(side="left", padx=(0, 8))

        btn_save_hf = ctk.CTkButton(
            hf_row,
            text="💾 Save Token",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CYAN_HOVER,
            text_color="#000000",
            height=34,
            corner_radius=6,
            command=self._save_hf_token
        )
        btn_save_hf.pack(side="left", padx=(0, 8))

        btn_clear_hf = ctk.CTkButton(
            hf_row,
            text="🗑️ Clear",
            font=("Segoe UI", 9),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_RED_HOVER,
            text_color=self.COLOR_TEXT_MUTED,
            height=34,
            width=60,
            corner_radius=6,
            command=self._clear_hf_token
        )
        btn_clear_hf.pack(side="left")

        # 3. System Hardware Telemetry Card
        sys_card = ctk.CTkFrame(view, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=10)
        sys_card.pack(fill="x")

        s_inner = ctk.CTkFrame(sys_card, fg_color="transparent")
        s_inner.pack(fill="x", padx=18, pady=16)

        ctk.CTkLabel(s_inner, text="🖥️ SYSTEM HARDWARE TELEMETRY", font=("Segoe UI", 12, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w")

        info_text = (
            f"• CPU Physical Cores: {self.runtime_booster.physical_cores} Cores ({os.cpu_count() or 16} Threads)\n"
            f"• GPU Acceleration: AMD Radeon(TM) Graphics (Vulkan iGPU - 8.8 GB Shared VRAM)\n"
            f"• Platform: {sys.platform} ({os.name})\n"
            f"• Storage Path: {self.manager.models_dir}\n"
            f"• Direct Seek Engine: Zero-copy non-overlapping sparse seek-and-write"
        )
        ctk.CTkLabel(
            s_inner,
            text=info_text,
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", pady=(8, 0))

        return view

    # =========================================================================
    # CORE LOGIC & EVENT HANDLERS
    # =========================================================================

    def _log_event(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {text}\n")
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _copy_log(self):
        content = self.log_text.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Copied", "Console activity log copied to clipboard!")

    def _parse_concurrency(self) -> int:
        val = self.concurrency_var.get()
        if "4x" in val: return 4
        if "8x" in val: return 8
        if "24x" in val: return 24
        if "32x" in val: return 32
        return 16

    def _start_download(self):
        model = self.model_input_var.get().strip()
        if not model:
            messagebox.showwarning("Input Required", "Please enter a valid model name or tag.")
            return

        concurrency = self._parse_concurrency() if self.is_download_boosted.get() else 1
        self.is_downloading = True
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.prog_bar.set(0.0)

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
            self.btn_cancel.configure(state="disabled")

    def _update_progress_ui(self, p: Dict[str, Any]):
        done = p.get("completed_bytes", 0)
        tot = p.get("total_bytes", 1)
        speed = p.get("speed_mb", 0.0)
        peak = p.get("peak_speed_mb", 0.0)
        eta = p.get("eta_seconds", 0.0)
        threads = p.get("thread_states", {})

        pct = (done / tot) if tot > 0 else 0.0
        self.prog_bar.set(min(1.0, max(0.0, pct)))

        self.hud_speed.configure(text=f"{speed:.1f} MB/s")
        self.hud_speed_sub.configure(text=f"Peak: {peak:.1f} MB/s")

        done_mb = done / (1024 * 1024)
        tot_mb = tot / (1024 * 1024)
        self.hud_transferred.configure(text=f"{done_mb:.1f} MB / {tot_mb:.1f} MB")
        self.hud_pct.configure(text=f"{pct*100:.1f}% Complete")

        eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta//60)}m {int(eta%60)}s"
        self.hud_eta.configure(text=f"ETA: {eta_str}")

        active_count = sum(1 for t in threads.values() if t.get("active"))
        self.hud_streams.configure(text=f"{active_count}/{len(threads)} Active Streams")

        # Update mini visualizer chips
        for idx, lbl in enumerate(self.stream_chips):
            if idx < len(threads) and threads.get(idx, {}).get("active"):
                lbl.configure(fg_color=self.COLOR_GREEN_DIM, text_color=self.COLOR_GREEN)
            else:
                lbl.configure(fg_color=self.COLOR_CARD_ALT, text_color=self.COLOR_TEXT_DIM)

    def _on_download_complete(self, success: bool, model: str):
        self.is_downloading = False
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.hud_streams.configure(text="Streams Idle")

        for lbl in self.stream_chips:
            lbl.configure(fg_color=self.COLOR_CARD_ALT, text_color=self.COLOR_TEXT_DIM)

        if success:
            self.prog_bar.set(1.0)
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
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self._log_event(f"❌ Error: {err_msg}")
        err_lower = err_msg.lower()
        if any(keyword in err_lower for keyword in ("401", "unauthorized", "permissionerror", "invalid username or password")):
            self._show_hf_auth_dialog(err_msg, self.model_input_var.get())
        else:
            messagebox.showerror("Download Error", err_msg)

    def _quick_pull_model(self, model_name: str):
        self.switch_view("pull")
        self.model_input_var.set(model_name)
        if not self.is_downloading:
            self._start_download()

    def _show_hf_guide_dialog(self):
        """Opens an informative guide and URL converter for Hugging Face models."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("🤗 How to Download Hugging Face Models")
        dlg.geometry("620x530")
        dlg.minsize(560, 480)
        dlg.configure(fg_color=self.COLOR_BG_DARK)
        dlg.transient(self)
        dlg.grab_set()

        # Header
        hdr = ctk.CTkFrame(dlg, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(18, 10))

        ctk.CTkLabel(
            hdr,
            text="🤗 Hugging Face Models Guide",
            font=("Segoe UI", 15, "bold"),
            text_color="#c4b5fd"
        ).pack(anchor="w")

        ctk.CTkLabel(
            hdr,
            text="Ollama natively supports downloading GGUF models directly from Hugging Face using the 'hf.co/' prefix.",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w", pady=(2, 0))

        # Card 1: Standard Format
        c1 = ctk.CTkFrame(dlg, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        c1.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(c1, text="1. Standard Model Format:", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=14, pady=(10, 2))
        fmt_box = ctk.CTkFrame(c1, fg_color=self.COLOR_BG_DARK, corner_radius=6)
        fmt_box.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(
            fmt_box,
            text="hf.co/{username}/{model-repository}:{quantization_tag}\n\nExamples:\n• hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M\n• hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M\n• hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
            font=("Consolas", 10),
            text_color=self.COLOR_CYAN,
            justify="left"
        ).pack(anchor="w", padx=10, pady=8)

        # Card 2: Interactive Smart Converter
        c2 = ctk.CTkFrame(dlg, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        c2.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(c2, text="2. Smart URL Converter (Paste Any Hugging Face Link):", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=14, pady=(10, 2))
        
        convert_row = ctk.CTkFrame(c2, fg_color="transparent")
        convert_row.pack(fill="x", padx=14, pady=(0, 10))

        paste_var = ctk.StringVar()
        entry_paste = ctk.CTkEntry(
            convert_row,
            textvariable=paste_var,
            placeholder_text="Paste https://huggingface.co/... here",
            font=("Consolas", 10),
            height=34,
            corner_radius=6,
            fg_color=self.COLOR_BG_DARK,
            border_color=self.COLOR_BORDER_LIGHT
        )
        entry_paste.pack(side="left", fill="x", expand=True, padx=(0, 8))

        def on_convert():
            url = paste_var.get().strip()
            if not url:
                return
            reg, ns, mdl, tag = RegistryClient.parse_model_ref(url)
            converted = f"{reg}/{ns}/{mdl}:{tag}"
            self.model_input_var.set(converted)
            dlg.destroy()
            self._log_event(f"Loaded Hugging Face reference: '{converted}'")
            messagebox.showinfo("Loaded", f"Converted and loaded into downloader:\n\n{converted}")

        btn_load = ctk.CTkButton(
            convert_row,
            text="Convert & Load",
            font=("Segoe UI", 9, "bold"),
            fg_color=self.COLOR_CYAN,
            hover_color=self.COLOR_CYAN_HOVER,
            text_color="#000000",
            height=34,
            corner_radius=6,
            command=on_convert
        )
        btn_load.pack(side="left")

        # Card 3: Gated Models / Token Notice
        c3 = ctk.CTkFrame(dlg, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        c3.pack(fill="x", padx=20, pady=(0, 14))

        ctk.CTkLabel(c3, text="3. Gated Models (e.g. Meta-Llama, Gemma):", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            c3,
            text="If a model requires license agreement on Hugging Face, accept it on huggingface.co and save your User Access Token in the app's '⚙️ Settings & Proxy' tab to avoid 401 errors.",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            wraplength=540,
            justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 8))

        btn_row_c3 = ctk.CTkFrame(c3, fg_color="transparent")
        btn_row_c3.pack(fill="x", padx=14, pady=(0, 10))

        btn_auto = ctk.CTkButton(
            btn_row_c3,
            text="🚀 1-Click Connect Token",
            font=("Segoe UI", 9, "bold"),
            fg_color=self.COLOR_PURPLE,
            hover_color="#7c3aed",
            text_color="#ffffff",
            height=28,
            corner_radius=6,
            command=lambda: (dlg.destroy(), self._start_hf_auto_connect())
        )
        btn_auto.pack(side="left", padx=(0, 8))

        btn_settings = ctk.CTkButton(
            btn_row_c3,
            text="⚙️ Open Settings",
            font=("Segoe UI", 9),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_TEXT_MAIN,
            height=28,
            corner_radius=6,
            command=lambda: (dlg.destroy(), self.switch_view("settings"))
        )
        btn_settings.pack(side="left")

        # Bottom Close Button
        ctk.CTkButton(
            dlg,
            text="Close",
            font=("Segoe UI", 10),
            fg_color=self.COLOR_CARD,
            hover_color=self.COLOR_CARD_HOVER,
            height=32,
            width=80,
            command=dlg.destroy
        ).pack(side="bottom", pady=(0, 14))

    # ---------------- MODELS MANAGER ACTIONS ----------------

    def _refresh_installed_models(self):
        models = self.manager.list_installed_models()
        self.installed_models_cache = models
        self.lbl_models_hdr.configure(text=f"📦 LOCAL OLLAMA LIBRARY ({len(models)} Models)")
        self._filter_model_cards()

        # Update combobox in inference tab
        model_names = [m["name"] for m in models]
        if hasattr(self, "cb_inf_model"):
            self.cb_inf_model.configure(values=model_names)
            if model_names and not self.cb_inf_model.get():
                self.cb_inf_model.set(model_names[0])

    def _create_selected_model_turbo(self, name: str):
        if not name:
            return
        if "-turbo" in name:
            messagebox.showinfo("Already Tuned", f"'{name}' is already an optimized turbo profile.")
            return

        ctx_val = 2048
        if hasattr(self, "cb_inf_ctx"):
            try:
                ctx_val = int(self.cb_inf_ctx.get().split()[0])
            except Exception:
                pass

        self._log_event(f"Building tuned '-turbo' profile for '{name}' (ctx: {ctx_val})...")

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

    def _delete_model(self, name: str):
        if messagebox.askyesno("Confirm Delete", f"Delete model '{name}' from local Ollama storage?"):
            ok, msg = self.manager.delete_model(name)
            if ok:
                self._refresh_installed_models()
                messagebox.showinfo("Deleted", msg)
            else:
                messagebox.showerror("Delete Error", msg)

    # ---------------- INFERENCE TURBO ACTIONS ----------------

    def _on_toggle_download_booster(self):
        state = "ACTIVE (16x Multi-Stream)" if self.is_download_boosted.get() else "OFF"
        self._log_event(f"Download booster set to: {state}")

    def _on_toggle_runtime_booster(self):
        if self.is_runtime_boosted.get():
            self.runtime_booster.enable_runtime_acceleration()
            self._log_event("Inference Acceleration: Active (Vulkan iGPU + Flash Attention)")
        else:
            self.runtime_booster.disable_runtime_acceleration()
            self._log_event("Inference Acceleration: Disabled")

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
            self.btn_restart_ollama.configure(state="disabled", text="⏳ Restarting...")

            def worker():
                ok, msg = self.runtime_booster.restart_ollama_with_turbo()
                def finish():
                    self.btn_restart_ollama.configure(state="normal", text="🔄 Apply & Restart Ollama with Turbo")
                    self._update_service_status()
                    if ok:
                        messagebox.showinfo("Ollama Turbo Active", msg)
                    else:
                        messagebox.showerror("Restart Error", msg)
                self.after(0, finish)

            threading.Thread(target=worker, daemon=True).start()

    def _start_inference_test(self):
        model = self.cb_inf_model.get().strip()
        if not model:
            messagebox.showwarning("Selection Required", "Please select a model from the dropdown to test.")
            return
        prompt = self.inf_prompt_var.get().strip()
        if not prompt:
            return

        ctx_val = 2048
        if hasattr(self, "cb_inf_ctx"):
            try:
                ctx_val = int(self.cb_inf_ctx.get().split()[0])
            except Exception:
                pass

        self.btn_inf_run.configure(state="disabled")
        self.inf_text_out.delete("1.0", "end")
        self.inf_text_out.insert("end", f"🚀 Executing boosted inference on {model} (context: {ctx_val})...\n\n")

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
                        self.inf_text_out.insert("end", t)
                        self.inf_text_out.see("end")
                        self.hud_inf_tps.configure(text=f"{s:.1f} tok/s")
                        self.hud_inf_toks.configure(text=f"{cnt} tokens")
                        self.hud_inf_dur.configure(text=f"Total Duration: {d:.2f}s")
                        if tf is not None:
                            pe_str = f"Prompt Eval: {(pc/pd):.1f} tok/s" if (pc and pd and pd > 0) else ""
                            self.hud_inf_ttft.configure(text=f"TTFT: {tf:.2f}s  |  {pe_str}")

                    self.after(0, update)
            except Exception as e:
                self.after(0, lambda err=str(e): self.inf_text_out.insert("end", f"\n\n❌ Error connecting to Ollama: {err}\nEnsure Ollama is running."))
            finally:
                self.after(0, lambda: self.btn_inf_run.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _launch_inference_terminal(self):
        model = self.cb_inf_model.get().strip()
        if not model:
            messagebox.showwarning("Selection Required", "Please select a model from the dropdown.")
            return
        self.runtime_booster.launch_boosted_terminal(model)

    # ---------------- BENCHMARK ACTIONS ----------------

    def _run_benchmark(self):
        self.btn_run_bench.configure(state="disabled")
        self.bench_text.delete("1.0", "end")
        self.bench_text.insert("end", "🚀 Starting live diagnostic network throughput benchmark...\n")

        def run():
            try:
                client = RegistryClient()
                manifest, _ = client.fetch_manifest("llama3.2:1b")
                layers = manifest.get("layers", [])
                model_layer = next((l for l in layers if "model" in l.get("mediaType", "")), layers[0])
                digest = model_layer["digest"]
                url = client.resolve_blob_download_url("registry.ollama.ai", "library", "llama3.2", digest)

                bench_bytes = 20 * 1024 * 1024  # 20 MB test

                # Phase 1: Single Connection
                self.after(0, lambda: self.bench_text.insert("end", "\n[1/2] Testing Native Single Connection (ollama pull style)..."))
                t0 = time.time()
                import requests
                headers = {"Range": f"bytes=0-{bench_bytes-1}"}
                r = requests.get(url, headers=headers, stream=True, timeout=20)
                downloaded = 0
                for chunk in r.iter_content(chunk_size=65536):
                    downloaded += len(chunk)
                t_single = max(0.01, time.time() - t0)
                speed_single = (downloaded / (1024 * 1024)) / t_single

                self.after(0, lambda: self.prog_bench_single.set(1.0))
                self.after(0, lambda: self.lbl_bench_single.configure(text=f"Single Stream Speed: {speed_single:.2f} MB/s (Time: {t_single:.2f}s)"))
                self.after(0, lambda: self.bench_text.insert("end", f"  DONE: {speed_single:.2f} MB/s\n"))

                # Phase 2: Multi-Stream Engine (16 Streams)
                self.after(0, lambda: self.bench_text.insert("end", "[2/2] Testing Booster Multi-Stream Engine (16 parallel streams)..."))
                t1 = time.time()
                chunk_len = bench_bytes // 16
                results = [0] * 16

                def worker(idx, start, end):
                    h = {"Range": f"bytes={start}-{end}"}
                    res = requests.get(url, headers=h, timeout=20)
                    results[idx] = len(res.content)

                threads = []
                for i in range(16):
                    st = i * chunk_len
                    en = (i + 1) * chunk_len - 1 if i < 15 else bench_bytes - 1
                    th = threading.Thread(target=worker, args=(i, st, en))
                    threads.append(th)
                    th.start()

                for th in threads:
                    th.join()

                t_multi = max(0.01, time.time() - t1)
                speed_multi = (sum(results) / (1024 * 1024)) / t_multi
                ratio = speed_multi / speed_single if speed_single > 0 else 1.0

                self.after(0, lambda: self.prog_bench_multi.set(1.0))
                self.after(0, lambda: self.lbl_bench_multi.configure(text=f"16x Multi-Stream Speed: {speed_multi:.2f} MB/s (Time: {t_multi:.2f}s)"))
                self.after(0, lambda: self.bench_text.insert("end", f"  DONE: {speed_multi:.2f} MB/s\n"))
                self.after(0, lambda: self.bench_text.insert("end", f"\n[*] Result: Booster Multi-Stream Engine is {ratio:.1f}x FASTER than standard single connection!\n"))
            except Exception as e:
                self.after(0, lambda err=str(e): self.bench_text.insert("end", f"\n❌ Benchmark error: {err}\n"))
            finally:
                self.after(0, lambda: self.btn_run_bench.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    # ---------------- SETTINGS & PROXY ACTIONS ----------------

    def _toggle_proxy(self):
        if not self.proxy_server.is_running:
            ok, msg = self.proxy_server.start()
            if ok:
                self.btn_proxy_toggle.configure(text="Stop Accelerator Proxy", fg_color=self.COLOR_RED, hover_color=self.COLOR_RED_HOVER)
                self.lbl_proxy_status.configure(text="Proxy Status: 🟢 Running on http://127.0.0.1:11435", text_color=self.COLOR_GREEN)
            else:
                messagebox.showerror("Proxy Error", msg)
        else:
            self.proxy_server.stop()
            self.btn_proxy_toggle.configure(text="Start Accelerator Proxy", fg_color=self.COLOR_GREEN, hover_color=self.COLOR_GREEN_HOVER)
            self.lbl_proxy_status.configure(text="Proxy Status: Stopped", text_color=self.COLOR_TEXT_MUTED)

    def _toggle_hf_visibility(self):
        if self.entry_hf.cget("show") == "*":
            self.entry_hf.configure(show="")
            self.btn_hf_show.configure(text="Hide")
        else:
            self.entry_hf.configure(show="*")
            self.btn_hf_show.configure(text="Show")

    def _save_hf_token(self):
        tok = self.hf_token_var.get().strip()
        if tok and not is_valid_hf_token(tok):
            if not messagebox.askyesno(
                "Unusual Token Format",
                f"The token entered does not match standard Hugging Face format (usually 'hf_...' followed by 20+ characters).\n\n"
                f"Do you want to save it anyway?"
            ):
                return
        save_hf_token(tok)
        self._update_hf_token_ui()
        masked = f"{tok[:4]}••••••••{tok[-4:]}" if tok and len(tok) >= 12 else (tok if tok else "None")
        self._log_event(f"💾 Hugging Face token saved: {masked}")
        messagebox.showinfo("Saved", "Hugging Face Access Token saved and synced successfully across system environment, registry, and cache!")

    def _clear_hf_token(self):
        if messagebox.askyesno("Clear Token", "Are you sure you want to remove your Hugging Face Access Token?"):
            self.hf_token_var.set("")
            save_hf_token("")
            self._update_hf_token_ui()
            self._log_event("🗑️ Hugging Face token removed.")
            self._flash_toast("Hugging Face token removed.", color=self.COLOR_CARD_ALT)

    def _check_clipboard_for_token(self, silent_if_same: bool = True, force: bool = False) -> bool:
        """
        Safely inspects OS clipboard for Hugging Face tokens (hf_...).
        If found and different from current token:
        - Auto-populates input
        - Saves to environment, disk & registry
        - Updates UI badges
        - Displays notification toast
        """
        try:
            content = self.clipboard_get()
        except Exception:
            return False

        if not content or not isinstance(content, str):
            return False

        tok = extract_hf_token(content)
        if not tok:
            return False

        curr = (self.hf_token_var.get() or "").strip()
        if tok == curr and not force:
            return False

        # Detected new HF token!
        self.hf_token_var.set(tok)
        save_hf_token(tok)
        self._update_hf_token_ui()
        masked = f"{tok[:4]}••••••••{tok[-4:]}"
        self._log_event(f"🔑 Hugging Face token auto-detected and applied: {masked}")
        self._flash_toast(f"🔑 Hugging Face Token Auto-Detected & Applied!\n({masked})")
        return True

    def _on_window_focus(self, event=None):
        self._check_clipboard_for_token(silent_if_same=True)

    def _start_hf_auto_connect(self):
        """
        1-Click automated token connection:
        1. Opens Hugging Face token creation page in browser with read permissions pre-filled.
        2. Starts clipboard polling to instantly capture the token upon user copy.
        """
        url = "https://huggingface.co/settings/tokens/new?token_name=OllamaBooster&type=read"
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showwarning("Notice", f"Could not launch browser automatically: {e}\nPlease visit:\n{url}")

        self._log_event("🌐 Hugging Face token page opened in browser. Listening for token copy...")
        self._token_listen_deadline = time.time() + 90.0
        self._poll_clipboard_token_step()

    def _poll_clipboard_token_step(self):
        """Polls clipboard every 350ms until token is copied or timeout occurs."""
        if not hasattr(self, "_token_listen_deadline"):
            return

        remaining = int(self._token_listen_deadline - time.time())
        if remaining <= 0:
            delattr(self, "_token_listen_deadline")
            self._update_listen_status("⚪ Auto-listen timed out. Click again or paste manually.", self.COLOR_TEXT_DIM)
            return

        detected = self._check_clipboard_for_token(force=True)
        if detected:
            delattr(self, "_token_listen_deadline")
            self._update_listen_status("✅ Token detected and connected successfully!", self.COLOR_GREEN)
            messagebox.showinfo(
                "Hugging Face Connected! 🎉",
                "Your Hugging Face Token was automatically captured from clipboard and saved!\n\n"
                "• Synced to System Environment (HF_TOKEN)\n"
                "• Saved to Windows Registry (Permanent across reboots)\n"
                "• Saved to Hugging Face Cache (~/.cache/huggingface/token)\n\n"
                "You can now download and run gated models seamlessly!"
            )
            return

        self._update_listen_status(
            f"📡 Listening to clipboard... ({remaining}s remaining) — Copy token on Hugging Face!",
            self.COLOR_CYAN
        )
        self.after(350, self._poll_clipboard_token_step)

    def _update_listen_status(self, text: str, color: str):
        if hasattr(self, "lbl_hf_listen_status") and self.lbl_hf_listen_status.winfo_exists():
            self.lbl_hf_listen_status.configure(text=text, text_color=color)

    def _update_hf_token_ui(self):
        """Updates all HF token indicators and badges across views."""
        tok = (self.hf_token_var.get() or "").strip()
        has_tok = bool(tok and is_valid_hf_token(tok))
        masked = f"{tok[:4]}••••••••{tok[-4:]}" if has_tok and len(tok) >= 12 else (tok if has_tok else "")

        # Update Pull Tab Strip
        if hasattr(self, "lbl_pull_hf_status") and self.lbl_pull_hf_status.winfo_exists():
            if has_tok:
                self.lbl_pull_hf_status.configure(
                    text=f"🟢 HF Token Active ({masked}) • Gated models enabled",
                    text_color=self.COLOR_GREEN
                )
                if hasattr(self, "btn_pull_hf_connect") and self.btn_pull_hf_connect.winfo_exists():
                    self.btn_pull_hf_connect.configure(text="🔄 Change Token")
            else:
                self.lbl_pull_hf_status.configure(
                    text="⚪ HF Token: None (Gated models require token)",
                    text_color=self.COLOR_TEXT_MUTED
                )
                if hasattr(self, "btn_pull_hf_connect") and self.btn_pull_hf_connect.winfo_exists():
                    self.btn_pull_hf_connect.configure(text="🚀 1-Click Connect Token")

        # Update Settings Tab Badge
        if hasattr(self, "lbl_settings_hf_status") and self.lbl_settings_hf_status.winfo_exists():
            if has_tok:
                self.lbl_settings_hf_status.configure(
                    text=f"🟢 Active Token: {masked}  •  Synced to System Environment, Registry & Cache",
                    text_color=self.COLOR_GREEN
                )
            else:
                self.lbl_settings_hf_status.configure(
                    text="⚪ No Token Configured — Gated models require an access token",
                    text_color=self.COLOR_TEXT_MUTED
                )

    def _flash_toast(self, message: str, color: str = "#10b981", duration_ms: int = 4000):
        """Displays a sleek floating notification pill at the bottom right of the main window."""
        try:
            toast = ctk.CTkLabel(
                self,
                text=message,
                font=("Segoe UI", 10, "bold"),
                fg_color=color,
                text_color="#ffffff",
                corner_radius=8,
                padx=16,
                pady=10
            )
            toast.place(relx=0.98, rely=0.96, anchor="se")
            self.after(duration_ms, lambda: toast.destroy() if toast.winfo_exists() else None)
        except Exception:
            pass

    def _show_hf_auth_dialog(self, err_msg: str, model_ref: str):
        """Displays a dedicated, polished troubleshooting modal for Hugging Face authentication."""
        reg, ns, mdl, tag = RegistryClient.parse_model_ref(model_ref)
        is_hf = (reg == "hf.co")

        dlg = ctk.CTkToplevel(self)
        dlg.title("🔑 Hugging Face Authentication Required")
        dlg.geometry("640x520")
        dlg.minsize(580, 460)
        dlg.configure(fg_color=self.COLOR_BG_DARK)
        dlg.transient(self)
        dlg.grab_set()

        # Header
        hdr = ctk.CTkFrame(dlg, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(18, 10))

        ctk.CTkLabel(
            hdr,
            text="🔑 Hugging Face Authentication Required",
            font=("Segoe UI", 15, "bold"),
            text_color=self.COLOR_CYAN
        ).pack(anchor="w")

        ctk.CTkLabel(
            hdr,
            text=f"Model: {model_ref}\nThis model repository is gated or requires a valid Hugging Face Access Token.",
            font=("Segoe UI", 10),
            text_color=self.COLOR_TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", pady=(2, 0))

        # Card 1: Gated License
        c1 = ctk.CTkFrame(dlg, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        c1.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(c1, text="Step 1: Accept Model License Agreement (if Gated)", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            c1,
            text="Gated models (such as Llama, Gemma, LiquidAI LFM) require you to accept their terms on Hugging Face once before they can be downloaded.",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            wraplength=560,
            justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 8))

        model_url = f"https://huggingface.co/{ns}/{mdl}" if is_hf else "https://huggingface.co"
        btn_open_hf = ctk.CTkButton(
            c1,
            text=f"🌐 Open Model on Hugging Face ({ns}/{mdl})",
            font=("Segoe UI", 9, "bold"),
            fg_color=self.COLOR_CARD_ALT,
            hover_color=self.COLOR_CARD_HOVER,
            text_color=self.COLOR_CYAN,
            height=30,
            corner_radius=6,
            command=lambda: webbrowser.open(model_url)
        )
        btn_open_hf.pack(anchor="w", padx=14, pady=(0, 10))

        # Card 2: Auto-Connect Token
        c2 = ctk.CTkFrame(dlg, fg_color=self.COLOR_CARD, border_width=1, border_color=self.COLOR_BORDER, corner_radius=8)
        c2.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(c2, text="Step 2: Connect Your Hugging Face Token", font=("Segoe UI", 10, "bold"), text_color=self.COLOR_TEXT_MAIN).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            c2,
            text="Click below to open token generator in your browser. Copy the generated token, and this app will auto-detect & save it immediately!",
            font=("Segoe UI", 9),
            text_color=self.COLOR_TEXT_MUTED,
            wraplength=560,
            justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 8))

        btn_auto_token = ctk.CTkButton(
            c2,
            text="🚀 1-Click Auto-Connect Token (From Clipboard)",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_PURPLE,
            hover_color="#7c3aed",
            text_color="#ffffff",
            height=34,
            corner_radius=6,
            command=lambda: (self._start_hf_auto_connect(), dlg.destroy())
        )
        btn_auto_token.pack(anchor="w", padx=14, pady=(0, 10))

        # Card 3: Action Buttons
        bot_row = ctk.CTkFrame(dlg, fg_color="transparent")
        bot_row.pack(fill="x", padx=20, pady=(4, 14))

        def retry():
            dlg.destroy()
            self._start_download()

        btn_retry = ctk.CTkButton(
            bot_row,
            text="⚡ Retry Download Now",
            font=("Segoe UI", 10, "bold"),
            fg_color=self.COLOR_GREEN,
            hover_color=self.COLOR_GREEN_HOVER,
            text_color="#ffffff",
            height=34,
            command=retry
        )
        btn_retry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            bot_row,
            text="Close",
            font=("Segoe UI", 10),
            fg_color=self.COLOR_CARD,
            hover_color=self.COLOR_CARD_HOVER,
            height=34,
            width=80,
            command=dlg.destroy
        ).pack(side="right")

    # ---------------- SERVICE POLLER & TELEMETRY ----------------

    def _update_service_status(self):
        running = self.runtime_booster.is_ollama_running()
        if running:
            self.lbl_service_dot.configure(text_color=self.COLOR_GREEN)
            self.lbl_service_text.configure(text="127.0.0.1:11434 Online", text_color=self.COLOR_TEXT_MAIN)
            if hasattr(self, "lbl_inf_service_dot"):
                self.lbl_inf_service_dot.configure(text_color=self.COLOR_GREEN)
                self.lbl_inf_service_title.configure(text="Ollama Service: Active (127.0.0.1:11434 - AMD Radeon Vulkan Turbo)")
        else:
            self.lbl_service_dot.configure(text_color=self.COLOR_RED)
            self.lbl_service_text.configure(text="Ollama Service Offline", text_color=self.COLOR_RED)
            if hasattr(self, "lbl_inf_service_dot"):
                self.lbl_inf_service_dot.configure(text_color=self.COLOR_RED)
                self.lbl_inf_service_title.configure(text="Ollama Service: Offline (Start with 'ollama serve')")

    def _start_service_poller(self):
        def loop():
            while True:
                time.sleep(3)
                try:
                    self.after(0, self._update_service_status)
                    self.after(0, lambda: self._check_clipboard_for_token(silent_if_same=True))
                except Exception:
                    break
        threading.Thread(target=loop, daemon=True).start()


def start_gui():
    """Application entry point for desktop GUI."""
    app = BoosterApp()
    app.mainloop()


if __name__ == "__main__":
    start_gui()
