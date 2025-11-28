#!/bin/bash
# Quick setup script for CTGAN notebook
# Run with: bash run_ctgan_setup.sh

echo "=========================================="
echo "CTGAN Notebook Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✓ Python found"
echo ""

# Create virtual environment
echo "2. Creating virtual environment..."
if [ ! -d "ctgan_env" ]; then
    python3 -m venv ctgan_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source ctgan_env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "5. Installing required packages..."
echo "   This may take a few minutes..."
pip install -r requirements_ctgan.txt --quiet
if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully"
else
    echo "ERROR: Package installation failed. Try manually with: pip install -r requirements_ctgan.txt"
    exit 1
fi
echo ""

# Verify CTGAN installation
echo "6. Verifying CTGAN installation..."
python -c "from ctgan import CTGAN; print('✓ CTGAN installed successfully')"
if [ $? -ne 0 ]; then
    echo "ERROR: CTGAN verification failed"
    exit 1
fi
echo ""

# Check for CSV files
echo "7. Checking for CSV data files..."
csv_files=("train_data_ads.csv" "train_data_feeds.csv" "test_data_ads.csv" "test_data_feeds.csv")
missing_files=()

for file in "${csv_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ Found: $file"
    else
        echo "   ✗ Missing: $file"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some CSV files are missing. The notebook requires:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please add these files to the project directory before running the notebook."
fi
echo ""

# Success message
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure your CSV files are in this directory"
echo "2. Launch Jupyter with: jupyter notebook"
echo "3. Open: ctgan_synthetic_data.ipynb"
echo "4. Run the cells!"
echo ""
echo "For detailed instructions, see: SETUP_CTGAN.md"
echo ""
echo "To activate the environment later, run:"
echo "   source ctgan_env/bin/activate"
echo ""
