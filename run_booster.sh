#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if command -v python3 >/dev/null 2>&1; then
    exec python3 main.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python main.py "$@"
else
    echo "Python is not installed or not in PATH."
    exit 1
fi
