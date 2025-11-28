# CTGAN Notebook Setup Guide

This guide will help you set up and run the `ctgan_synthetic_data.ipynb` notebook locally on your machine.

## Prerequisites

You need:
- Python 3.8 or higher
- The CSV data files in the project directory (✓ You already have these!)

## Step-by-Step Setup

### 1. Check Your Python Version

Open a terminal in the `Click_Through_Pred` folder and run:

```bash
python --version
# or
python3 --version
```

**Required:** Python 3.8 or higher

---

### 2. Create a Virtual Environment (Recommended)

This keeps your dependencies organized and prevents conflicts:

```bash
# Create virtual environment
python3 -m venv ctgan_env

# Activate it
# On macOS/Linux:
source ctgan_env/bin/activate

# On Windows:
ctgan_env\Scripts\activate
```

You should see `(ctgan_env)` in your terminal prompt when activated.

---

### 3. Install Required Packages

Install all dependencies in one go:

```bash
pip install --upgrade pip

pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook ctgan
```

**Key packages:**
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` & `seaborn` - Visualization
- `scikit-learn` - Machine learning (logistic regression, metrics)
- `jupyter notebook` - Notebook interface
- `ctgan` - The synthetic data generation library

**Expected install time:** 2-5 minutes depending on your internet connection

---

### 4. Verify CTGAN Installation

Check that CTGAN installed correctly:

```bash
python -c "from ctgan import CTGAN; print('CTGAN installed successfully!')"
```

You should see: `CTGAN installed successfully!`

---

### 5. Launch Jupyter Notebook

Start Jupyter from your project directory:

```bash
jupyter notebook
```

This will:
1. Start the Jupyter server
2. Open your browser automatically
3. Show a file browser with your notebooks

**If the browser doesn't open automatically:**
- Look for a URL in the terminal output (e.g., `http://localhost:8888/?token=...`)
- Copy and paste it into your browser

---

### 6. Open the CTGAN Notebook

In the Jupyter file browser:
1. Click on `ctgan_synthetic_data.ipynb`
2. The notebook will open in a new tab

---

### 7. Run the Notebook

You can run the notebook in two ways:

**Option A: Run All Cells at Once**
- Click `Cell` → `Run All` from the menu
- Or use keyboard shortcut: `Shift + Enter` repeatedly

**Option B: Run Step-by-Step (Recommended for first run)**
- Click on the first code cell
- Press `Shift + Enter` to run it and move to the next
- Review output after each cell
- Continue through all cells

**Expected runtime:**
- Data loading and preprocessing: 5-10 minutes
- CTGAN training (100 epochs): 15-30 minutes
- Synthetic generation: 5-10 minutes
- Total: ~30-60 minutes depending on your machine

---

## What to Expect

### Section-by-Section Guide

1. **Import Libraries** (Cell 1)
   - Should run quickly
   - If errors: missing packages, see troubleshooting below

2. **Load and Merge Data** (Cells 2-4)
   - Loads your CSV files
   - Expected output: Dataset shapes and merge confirmation
   - **Verify:** You should see ~7.6M rows after merging

3. **Verify Data Structure** (Cells 5-6)
   - Shows class distribution (should be ~98.5% no-clicks, 1.5% clicks)
   - Feature cardinality analysis

4. **Target Encoding** (Cells 7-8)
   - Encodes high-cardinality features (task_id, device_size)
   - Watch for encoded column ranges

5. **Grouping Rare Categories** (Cells 9-10)
   - Groups adv_prim_id and city
   - Reduces cardinality for CTGAN

6. **Prepare CTGAN Data** (Cells 11-12)
   - Separates clicks from no-clicks
   - Calculates synthetic samples needed (~971K)

7. **Install and Train CTGAN** (Cells 13-15)
   - **This is the longest step!** (15-30 minutes)
   - You'll see training progress with epoch numbers
   - Model saves to `ctgan_model.pkl`

8. **Generate Synthetic Data** (Cells 16-18)
   - Generates synthetic clicks
   - Compares distributions (real vs synthetic)

9. **Combine and Encode** (Cells 19-21)
   - Merges real + synthetic data
   - Applies encoding pipeline
   - Final dataset: 85% no-clicks, 15% clicks

10. **Train and Evaluate** (Cells 22-25)
    - Trains logistic regression
    - Shows performance metrics
    - Compares against baseline models

11. **Save Results** (Cell 26)
    - Saves trained models and results

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'ctgan'`

**Solution:**
```bash
pip install ctgan
```

### Problem: `FileNotFoundError: train_data_ads.csv not found`

**Solution:**
- Verify CSV files are in the same folder as the notebook
- Check file names match exactly (case-sensitive!)
- Run `ls *.csv` in terminal to list CSV files

### Problem: Jupyter won't start

**Solution:**
```bash
# Install/reinstall Jupyter
pip install --upgrade jupyter notebook

# Try starting again
jupyter notebook
```

### Problem: Out of Memory during CTGAN training

**Solution:**
- Reduce batch size in the CTGAN initialization (Cell 13):
  ```python
  ctgan = CTGAN(
      epochs=100,
      batch_size=250,  # Reduced from 500
      ...
  )
  ```

### Problem: CTGAN training is too slow

**Solutions:**
- Reduce epochs (e.g., from 100 to 50) for faster training
- Close other applications to free up memory
- Training on CPU is normal but slower than GPU

### Problem: Kernel crashes or freezes

**Solution:**
- Restart kernel: `Kernel` → `Restart` from menu
- Clear outputs: `Cell` → `All Output` → `Clear`
- Run cells again from the beginning

---

## Files That Will Be Created

After running the notebook, you'll have:

```
Click_Through_Pred/
├── ctgan_model.pkl              # Trained CTGAN model (~50-100 MB)
├── lr_ctgan_model.pkl           # Trained logistic regression model
├── ctgan_scaler.pkl             # Feature scaler
├── encoding_maps.pkl            # Encoding mappings
├── model_comparison_ctgan.csv   # Performance comparison table
└── ctgan_synthetic_data.ipynb   # The notebook (with outputs)
```

**Note:** These files are gitignored and won't be committed to your repository.

---

## Expected Results

You should see performance metrics like:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| LR (Downsampled) | 0.9765 | 0.1216 | 0.0822 | 0.0981 | 0.7490 |
| LR (Class Weighted) | 0.6419 | 0.0306 | 0.7194 | 0.0587 | 0.7499 |
| LR (CTGAN Synthetic) | ? | ? | ? | ? | ? |

The CTGAN model aims to balance precision and recall better than the baselines.

---

## Next Steps After Running

1. **Review the results** - Compare CTGAN model against baselines
2. **Experiment with parameters:**
   - Try different epoch counts (50, 150, 200)
   - Adjust target ratio (90-10, 80-20 instead of 85-15)
   - Modify batch size for performance tuning
3. **Save important outputs** - Export visualizations and metrics
4. **Document findings** - Note which approach works best for your use case

---

## Need Help?

If you encounter issues not covered here:
1. Check the error message carefully
2. Verify all CSV files are present and accessible
3. Ensure virtual environment is activated
4. Check that all packages installed successfully

---

**You're ready to go! Start with Step 1 above and work through the setup.**

Good luck with your CTGAN synthetic data generation! 🚀
