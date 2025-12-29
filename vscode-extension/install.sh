#!/bin/bash
# Install script for HIP/Triton Generator VSCode Extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "HIP/Triton Generator - VSCode Extension Installer"
echo "================================================"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js 18+ first"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    exit 1
fi

echo ""
echo "1. Installing npm dependencies..."
cd "$SCRIPT_DIR"
npm install

echo ""
echo "2. Compiling TypeScript..."
npm run compile

echo ""
echo "3. Checking for VSCode/Cursor..."

# Install the extension
if command -v code &> /dev/null; then
    echo "   Found 'code' command, installing extension..."
    code --install-extension "$SCRIPT_DIR" --force 2>/dev/null || true
elif command -v cursor &> /dev/null; then
    echo "   Found 'cursor' command, installing extension..."
    cursor --install-extension "$SCRIPT_DIR" --force 2>/dev/null || true
else
    echo "   VSCode/Cursor CLI not found."
    echo "   To install manually:"
    echo "   1. Open VSCode/Cursor"
    echo "   2. Go to Extensions (Ctrl+Shift+X)"
    echo "   3. Click '...' -> 'Install from VSIX'"
    echo "   4. Or run: npm run package && code --install-extension *.vsix"
fi

echo ""
echo "================================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Reload VSCode/Cursor (Ctrl+Shift+P -> 'Reload Window')"
echo "2. Configure your AMD API Key:"
echo "   - Open Settings (Ctrl+,)"
echo "   - Search for 'HIP Generator'"
echo "   - Set 'hipGenerator.amdApiKey'"
echo ""
echo "3. Usage:"
echo "   - Select PyTorch code in a .py file"
echo "   - Right-click -> 'Generate Triton Kernel'"
echo "================================================"











