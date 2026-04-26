# %%
# Klasifikasi Orange dan Grapefruit

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

# %%
DATA_PATH = "citrus.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
df = pd.read_csv(DATA_PATH)

print("Shape dataset:", df.shape)
print(df.head())
print("\nDistribusi kelas:")
print(df["name"].value_counts())

# %%
X = df.drop(columns=["name"])
y = LabelEncoder().fit_transform(df["name"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# %%
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}

# %%
def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Class 0", "Class 1"])
    plt.yticks([0, 1], ["Class 0", "Class 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_prob, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{title} (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

# %%
summary = []

for model_name, model in models.items():
    print(f"\n {model_name} ")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_prob)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(auc_value, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    summary.append([model_name, acc, prec, rec, f1, auc_value])

    safe_name = model_name.replace(" ", "_").lower()
    plot_confusion_matrix(
        y_test,
        y_pred,
        f"Confusion Matrix - {model_name}",
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png")
    )

    plot_precision_recall_curve(
        y_test,
        y_prob,
        model_name,
        os.path.join(OUTPUT_DIR, f"precision_recall_{safe_name}.png")
    )

    plot_roc_curve(
        y_test,
        y_prob,
        model_name,
        os.path.join(OUTPUT_DIR, f"roc_curve_{safe_name}.png")
    )

# %%
result_df = pd.DataFrame(
    summary,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
)

print("\n Ringkasan Hasil ")
print(result_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

result_df.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)
