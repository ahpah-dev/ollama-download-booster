#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) > 1:
        from .cli import cli_main
        cli_main()
    else:
        from .gui import start_gui
        start_gui()

if __name__ == "__main__":
    main()
