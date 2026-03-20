# Heart Disease Prediction ML Analysis

This project is a Python data science project that performs **Exploratory Data Analysis (EDA)** and trains **three machine learning models** to predict heart disease. This includes built-in synthetic data so anyone can run it instantly, meaning no dataset download is required.

---

##  What It Does

| Step | Description | Output |
|------|-------------|--------|
| EDA | Distribution plots for key clinical features | `outputs/01_eda.png` |
| Correlation | Feature correlation heatmap | `outputs/02_correlation.png` |
| ML Models | Logistic Regression, Random Forest, Gradient Boosting | — |
| ROC Curves | Model comparison by AUC score | `outputs/03_roc_curves.png` |
| Confusion Matrix | Per-model prediction accuracy breakdown | `outputs/04_confusion.png` |
| Feature Importance | Top predictive features (Random Forest) | `outputs/05_feature_importance.png` |
| Report | Full metrics summary | `outputs/summary_report.txt` |

---

## How to start on Windows, Mac, Linux

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/biomedical-ml-analysis.git
cd biomedical-ml-analysis
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analysis
```bash
# Using built-in synthetic data (600 patients)
python analyze.py

# Using your own CSV file
python analyze.py --csv path/to/your_data.csv
```

Results will appear in the `outputs/` folder.

---

## Project Structure

```
biomedical-ml-analysis/
│
├── analyze.py            <- Main script (run this)
├── requirements.txt      <- Python dependencies
├── README.md             <- You are here
│
├── data/                 <- Drop your own CSV files here (optional)
│
└── outputs/             <- Auto-generated after running
    ├── 01_eda.png
    ├── 02_correlation.png
    ├── 03_roc_curves.png
    ├── 04_confusion.png
    ├── 05_feature_importance.png
    └── summary_report.txt
```

---

## CSV Format (if using your own data)

Your CSV must include these columns (standard Cleveland Heart Disease column names are auto-renamed):

| Column | Description | Values |
|--------|-------------|--------|
| `age` | Patient age | integer |
| `sex` | Sex | 0 = female, 1 = male |
| `chest_pain_type` | CP type | 0–3 |
| `resting_bp` | Resting BP (mmHg) | integer |
| `cholesterol` | Serum cholesterol (mg/dl) | integer |
| `fasting_blood_sugar` | FBS > 120 mg/dl | 0 or 1 |
| `rest_ecg` | Resting ECG results | 0–2 |
| `max_heart_rate` | Max heart rate achieved | integer |
| `exercise_angina` | Exercise induced angina | 0 or 1 |
| `st_depression` | ST depression | float |
| `slope` | Slope of peak ST segment | 0–2 |
| `num_vessels` | # of major vessels (0–3) | 0–3 |
| `thal` | Thalassemia | 1, 2, or 3 |
| `heart_disease` | **Target** — 0 = No, 1 = Yes | 0 or 1 |

> The script also accepts the raw UCI Cleveland column names (`trestbps`, `chol`, `thalach`, etc.) and renames them automatically.

---

## The Models Used

- **Logistic Regression** - interpretable baseline
- **Random Forest** - ensemble method, used for feature importance
- **Gradient Boosting** - typically highest AUC

All models are evaluated with:
- 5-fold stratified cross-validation
- ROC-AUC, Accuracy, Precision, Recall, F1

---

## Sample Output

After running, you'll see charts like these saved to `outputs/`:

- Dark-themed distribution histograms by disease status
- Full feature correlation heatmap
- Three overlaid ROC curves
- Side-by-side confusion matrices
- Horizontal bar chart of feature importances

---

## Requirements to Run Project

- Python 3.9+
- See `requirements.txt` for packages

---

## License

MIT - free to use, modify, and share.
