#!/bin/bash

# Script to prepare the development environment
# Sets up environment variables and git hooks

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
    else
        echo "⚠️  .env.example not found, skipping environment file setup"
    fi
else
    echo "✅ .env already exists"
fi

echo "📝 Please edit .env as needed for your configuration."

# Install git hooks if they exist
if [ -d scripts/hooks ]; then
    for hook in scripts/hooks/*; do
        if [ -f "$hook" ]; then
            hook_name=$(basename "$hook")
            cp "$hook" ".git/hooks/$hook_name"
            chmod +x ".git/hooks/$hook_name"
            echo "✅ Installed hook: $hook_name"
        fi
    done
else
    echo "⚠️  scripts/hooks directory not found, skipping git hooks setup"
fi

echo ""
echo "🎉 Environment preparation complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env if needed"
echo "  2. Create a virtual environment: python -m venv venv"
echo "  3. Activate it: source venv/bin/activate"
echo "  4. Install dependencies: pip install -r requirements.txt"
echo "  5. Run the app: streamlit run app.py"
