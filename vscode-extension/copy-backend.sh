#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
DST_DIR="$SCRIPT_DIR/hipgenerator"

echo "Copying HipGenerator Python source..."
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"

# Create destination directory
mkdir -p "$DST_DIR"
mkdir -p "$DST_DIR/prompts"
mkdir -p "$DST_DIR/tools"

# Copy main Python files
cp "$SRC_DIR/generate.py" "$DST_DIR/"
cp "$SRC_DIR/eval.py" "$DST_DIR/"
cp "$SRC_DIR/run_loop.py" "$DST_DIR/"
cp "$SRC_DIR/requirements.txt" "$DST_DIR/"

# Copy prompts directory
cp "$SRC_DIR/prompts/"*.txt "$DST_DIR/prompts/" 2>/dev/null || true

# Copy tools directory
cp "$SRC_DIR/tools/"*.py "$DST_DIR/tools/" 2>/dev/null || true

echo "Done! Contents of $DST_DIR:"
ls -la "$DST_DIR"

