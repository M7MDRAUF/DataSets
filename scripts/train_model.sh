#!/bin/bash
# CineMatch V2.1.6 - Model Training Script
# Wrapper script for training the SVD model

echo "========================================="
echo "CineMatch V2.1.6 - Model Training"
echo "=========================================""
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

# Check if virtual environment is activated (recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   It's recommended to use a virtual environment"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the training script
echo "🚀 Starting model training..."
echo ""

python src/model_training.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Training completed successfully!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run the app: streamlit run app/main.py"
    echo "2. Or use Docker: docker-compose up"
else
    echo ""
    echo "========================================="
    echo "❌ Training failed!"
    echo "========================================="
    echo ""
    echo "Please check the error messages above"
    exit 1
fi
