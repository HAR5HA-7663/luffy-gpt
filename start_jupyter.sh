#!/bin/bash

# Start Jupyter Notebook for the Luffy GPT project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/venv/bin/activate"

echo "Starting Jupyter Notebook..."
echo "URL: http://localhost:8888/?token=har5ha"
echo "Press Ctrl+C to stop."

jupyter notebook --notebook-dir="$SCRIPT_DIR"
