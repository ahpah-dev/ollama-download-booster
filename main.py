#!/usr/bin/env python3
"""
Ollama Download Booster - Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Launches Desktop GUI or CLI depending on command line arguments.
"""

import sys

def main():
    if len(sys.argv) > 1:
        from ollama_booster.cli import cli_main
        cli_main()
    else:
        from ollama_booster.gui import start_gui
        start_gui()

if __name__ == "__main__":
    main()
