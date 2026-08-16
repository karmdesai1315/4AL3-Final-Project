# Diabetes Binary Classifier (SVM & Random Forest)

A Python script that trains and evaluates two machine learning models. A
Support Vector Machine (SVM) and a Random Forest Classifier (RFC). Using both, it will
predict whether an individual has diabetes/pre-diabetes based on health
survey indicators from the BRFSS 2015 dataset.

## Features

- **Feature selection** using `SelectKBest` with mutual information to pick
  the 10 most informative health indicators out of a larger candidate set
  (e.g. BMI, blood pressure, cholesterol, physical activity, general health).
- **Hyperparameter tuning** for both models via `GridSearchCV`:
  - SVM: tunes `C`, `kernel`, and `gamma`
  - RFC: tunes `n_estimators` and `class_weight`
- **Stratified 5-fold cross-validation** for improved performance estimates.
- **Custom evaluation metrics**, including:
  - True Skill Statistic (TSS)
  - Hinge loss
  - Precision, recall, F1-score, accuracy
  - Averaged confusion matrix across folds
- **Baseline comparison** using a majority-class (Boyer-Moore-based)
  predictor.
- **Interactive mode**: at runtime, choose to either
  1. Generate result graphs (confusion matrices + TSS score bar chart), or
  2. Enter your own feature values to get a live prediction with confidence.

## Requirements

- Python 3.12
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset

The script expects a CSV file named:
```
diabetes_binary_health_indicators_BRFSS2015.csv
```
placed in the same directory as the script. Only the first 1000 rows are
read by default (`nrows=1000`).

The following columns are dropped before modeling: `AnyHealthcare`,
`NoDocbcCost`, `MentHlth`, `DiffWalk`, `Education`, `Income`.

## Usage

Run the script directly:
```bash
python diabetes_classifier.py
```

You'll be prompted:
```
Would you like to generate graphs(1) or predict the output from one set of values(2)?
```

- **Option 1**: Trains both models, runs cross-validation, and displays:
  - Confusion matrix for the SVM model
  - Confusion matrix for the RFC model
  - A bar chart of TSS scores across each fold (with mean and standard
    deviation)
- **Option 2**: Prompts you to enter a value for each selected feature, then
  outputs a prediction (diabetes/pre-diabetes vs. none) along with the
  model's confidence percentage.

## Output

For both models, the script prints to console:
- Best hyperparameters found via grid search
- Average TSS (SVM only), hinge loss, precision, recall, F1-score, and
  accuracy across the 5 folds

## Project Structure

```
.
├── diabetes_classifier.py   # Main script (SVM + RFC classes and experiment())
└── diabetes_binary_health_indicators_BRFSS2015.csv   # Input dataset (not included)
```

## Notes

- Feature scaling is performed with `StandardScaler` before training.
- Class imbalance is handled via `class_weight='balanced'` (SVM) or
  `'balanced'` / `'balanced_subsample'` (RFC, tuned via grid search).
- Random forest and cross-validation splits use `random_state=42` for
  reproducibility.
