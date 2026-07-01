import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    words = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
    return " ".join(words)


def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(preprocess)
    return df


def build_features(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]
    return X, y, vectorizer


def train_models(X_train, y_train):
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    results = []
    confusion_matrices = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})
        confusion_matrices[name] = confusion_matrix(y_test, preds, labels=model.classes_)
    return pd.DataFrame(results), confusion_matrices


def cross_validate_models(models, X, y, cv=5):
    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        scores[name] = cv_scores.mean()
    return scores


def tune_svm(X_train, y_train):
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring="accuracy")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def plot_comparison(results_df, out_path="comparison_chart.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_confusion_matrices(confusion_matrices, labels, out_path="confusion_matrices.png"):
    fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(5 * len(confusion_matrices), 4))
    if len(confusion_matrices) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_best_model(models, results_df, vectorizer, path="best_model.pkl"):
    best_name = results_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    joblib.dump({"model": best_model, "vectorizer": vectorizer, "name": best_name}, path)
    return best_name, path


def predict_all(text, models, vectorizer):
    clean = preprocess(text)
    X = vectorizer.transform([clean])
    predictions = {name: model.predict(X)[0] for name, model in models.items()}
    return predictions


def run_cli(models, vectorizer, best_model_name):
    while True:
        text = input("\nEnter text (or 'quit' to exit): ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        predictions = predict_all(text, models, vectorizer)
        for name, pred in predictions.items():
            print(f"{name}: {pred}")
        print(f"Best Model ({best_model_name}) Prediction: {predictions[best_model_name]}")


def main():
    df = load_data()
    X, y, vectorizer = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = train_models(X_train, y_train)
    results_df, confusion_matrices = evaluate_models(models, X_test, y_test)
    print(results_df.to_string(index=False))

    cv_scores = cross_validate_models(models, X, y)
    print("\nCross-validation accuracy:")
    for name, score in cv_scores.items():
        print(f"{name}: {score:.3f}")

    best_svm, best_params = tune_svm(X_train, y_train)
    print("\nBest SVM params:", best_params)
    models["SVM"] = best_svm
    results_df, confusion_matrices = evaluate_models(models, X_test, y_test)

    plot_comparison(results_df)
    plot_confusion_matrices(confusion_matrices, labels=sorted(y.unique()))

    best_name, path = save_best_model(models, results_df, vectorizer)
    print(f"\nBest model: {best_name}, saved to {path}")

    run_cli(models, vectorizer, best_name)


if __name__ == "__main__":
    main()
