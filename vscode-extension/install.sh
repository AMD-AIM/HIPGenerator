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
echo "3. Installing vsce (VSCode packaging tool)..."
npm install -g @vscode/vsce 2>/dev/null || npm install @vscode/vsce

echo ""
echo "4. Packaging extension as VSIX..."
# Clean up old vsix files
rm -f "$SCRIPT_DIR"/*.vsix

# Package the extension
npx vsce package --allow-missing-repository
VSIX_FILE=$(ls "$SCRIPT_DIR"/*.vsix 2>/dev/null | head -1)

if [ -z "$VSIX_FILE" ]; then
    echo "Error: Failed to create VSIX package"
    exit 1
fi
echo "Created: $VSIX_FILE"

echo ""
echo "5. Installing extension..."

# Try different CLI tools
INSTALLED=false

# Try cursor CLI (remote server)
if [ -f "/root/.cursor-server/bin/"*"/bin/remote-cli/cursor" ]; then
    CURSOR_CLI=$(ls /root/.cursor-server/bin/*/bin/remote-cli/cursor 2>/dev/null | head -1)
    if [ -n "$CURSOR_CLI" ]; then
        echo "   Using Cursor remote CLI: $CURSOR_CLI"
        "$CURSOR_CLI" --install-extension "$VSIX_FILE" --force && INSTALLED=true
    fi
fi

# Try cursor command
if [ "$INSTALLED" = false ] && command -v cursor &> /dev/null; then
    echo "   Using cursor command..."
    cursor --install-extension "$VSIX_FILE" --force && INSTALLED=true
fi

# Try code command
if [ "$INSTALLED" = false ] && command -v code &> /dev/null; then
    echo "   Using code command..."
    code --install-extension "$VSIX_FILE" --force && INSTALLED=true
fi

if [ "$INSTALLED" = false ]; then
    echo ""
    echo "   ⚠ Could not auto-install. Please install manually:"
    echo "   1. Open VSCode/Cursor"
    echo "   2. Press Ctrl+Shift+P"
    echo "   3. Type 'Extensions: Install from VSIX'"
    echo "   4. Select: $VSIX_FILE"
fi

echo ""
echo "================================================"
echo "Installation complete!"
echo ""
echo "VSIX file location: $VSIX_FILE"
echo ""
echo "Next steps:"
echo "1. Reload VSCode/Cursor window:"
echo "   - Press Ctrl+Shift+P"
echo "   - Type 'Reload Window' and press Enter"
echo ""
echo "2. Configure your AMD API Key:"
echo "   - Open Settings (Ctrl+,)"
echo "   - Search for 'HIP Generator'"
echo "   - Set 'hipGenerator.amdApiKey' to your key"
echo ""
echo "3. Usage:"
echo "   - Open a .py file"
echo "   - Select PyTorch code"
echo "   - Right-click -> 'Generate Triton Kernel'"
echo "   - Or Right-click -> 'Optimize Triton Kernel'"
echo "================================================"
