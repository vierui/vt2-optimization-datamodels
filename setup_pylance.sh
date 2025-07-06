#!/bin/bash

# Setup script for Pylance memory optimization in Cursor

echo "Setting up Pylance memory optimization..."

# Add NODE_OPTIONS to your shell profile
SHELL_PROFILE=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    echo "Adding NODE_OPTIONS to $SHELL_PROFILE"
    echo "" >> "$SHELL_PROFILE"
    echo "# Pylance memory optimization" >> "$SHELL_PROFILE"
    echo "export NODE_OPTIONS=\"--max-old-space-size=8192\"" >> "$SHELL_PROFILE"
    echo "Added NODE_OPTIONS to $SHELL_PROFILE"
    echo "Please restart your terminal or run: source $SHELL_PROFILE"
else
    echo "Could not detect shell profile. Please manually add:"
    echo "export NODE_OPTIONS=\"--max-old-space-size=8192\""
    echo "to your shell configuration file."
fi

echo "Setup complete!"
echo ""
echo "Additional steps:"
echo "1. Restart Cursor"
echo "2. If issues persist, install Node.js 18+ separately"
echo "3. Update .vscode/settings.json with the path to your Node.js executable" 