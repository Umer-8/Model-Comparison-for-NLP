# Model Comparison NLP — Naive Bayes vs SVM vs Logistic Regression

## Files
- `dataset.csv` — 200 labeled samples (positive/negative), synthetic sentiment data
- `train_models.py` — full pipeline: preprocessing, TF-IDF, training, evaluation, comparison, tuning, CLI
- `best_model.pkl` — saved best model + vectorizer (generated after running)
- `comparison_chart.png`, `confusion_matrices.png` — generated after running

## Run
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
python train_models.py
```

Prints the comparison table, cross-validation scores, tuned SVM params, saves plots and the
best model, then drops into a CLI prompt where you can type text and get predictions from
all three models plus the best model's verdict.
