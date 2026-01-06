#!/bin/bash
# Manual download script for Meta-Tool datasets
# Run this script in an environment with internet access

set -e

echo "=========================================="
echo "Meta-Tool Dataset Download Script"
echo "=========================================="

# Create directories
mkdir -p data/gorilla
mkdir -p data/spider

# Download Gorilla APIBench
echo ""
echo "Downloading Gorilla APIBench..."
cd data/gorilla

# Gorilla evaluation files
BASE_URL="https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench"
for file in huggingface_eval.json tensorflow_eval.json torchhub_eval.json; do
    echo "  Downloading $file..."
    curl -fsSL "$BASE_URL/$file" -o "$file" || wget -q "$BASE_URL/$file" -O "$file" || echo "  Failed to download $file"
done

cd ../..

# Download Spider (if available)
echo ""
echo "Spider dataset can be downloaded from: https://yale-lily.github.io/spider"
echo "Or use: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('spider')\""

# Summary
echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Files downloaded to:"
echo "  - data/gorilla/"
echo ""
echo "The ToolBench dataset will be downloaded automatically"
echo "from HuggingFace when you run the training script."
echo ""
echo "To verify, run:"
echo "  python -c \"from data_loader import setup_all_datasets; setup_all_datasets()\""
