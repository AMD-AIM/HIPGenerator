#!/bin/bash
# Obfuscate Python source code before packaging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
DST_DIR="$SCRIPT_DIR/hipgenerator"
TEMP_DIR="$SCRIPT_DIR/.obfuscate_temp"

echo "=== AMD HIP/Triton Generator - Code Obfuscation ==="
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"

# Clean up
rm -rf "$DST_DIR" "$TEMP_DIR"
mkdir -p "$DST_DIR" "$DST_DIR/prompts" "$DST_DIR/tools" "$TEMP_DIR"

# Copy source files to temp directory
cp "$SRC_DIR/generate.py" "$TEMP_DIR/"
cp "$SRC_DIR/eval.py" "$TEMP_DIR/"
cp "$SRC_DIR/run_loop.py" "$TEMP_DIR/"

# Copy tools
mkdir -p "$TEMP_DIR/tools"
cp "$SRC_DIR/tools/"*.py "$TEMP_DIR/tools/" 2>/dev/null || true

echo "Obfuscating Python files..."

# Use pyarmor to obfuscate
cd "$TEMP_DIR"
pyarmor gen --output "$DST_DIR" generate.py eval.py run_loop.py 2>/dev/null || {
    echo "Warning: pyarmor failed, falling back to bytecode compilation"
    # Fallback: compile to .pyc
    python3 -m py_compile generate.py
    python3 -m py_compile eval.py
    python3 -m py_compile run_loop.py
    
    # Move compiled files
    mv __pycache__/*.pyc "$DST_DIR/" 2>/dev/null || true
    
    # Also copy source as backup (will be replaced by loader)
    cp generate.py eval.py run_loop.py "$DST_DIR/"
}

# Obfuscate tools
if [ -d "$TEMP_DIR/tools" ] && [ "$(ls -A $TEMP_DIR/tools/*.py 2>/dev/null)" ]; then
    cd "$TEMP_DIR/tools"
    pyarmor gen --output "$DST_DIR/tools" *.py 2>/dev/null || {
        echo "Warning: tools obfuscation failed, compiling to bytecode"
        for f in *.py; do
            python3 -m py_compile "$f" 2>/dev/null || true
        done
        mv __pycache__/*.pyc "$DST_DIR/tools/" 2>/dev/null || true
        cp *.py "$DST_DIR/tools/"
    }
fi

# Copy non-Python files (prompts, requirements)
cp "$SRC_DIR/requirements.txt" "$DST_DIR/"
cp "$SRC_DIR/prompts/"*.txt "$DST_DIR/prompts/" 2>/dev/null || true

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Obfuscation Complete ==="
ls -la "$DST_DIR"

