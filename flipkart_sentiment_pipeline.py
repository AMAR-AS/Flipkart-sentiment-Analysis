#!/usr/bin/env python3
"""
Flipkart Reviews Sentiment Analysis - Full pipeline
Usage: set DATA_PATH, then: python flipkart_sentiment_pipeline.py
"""
import warnings
warnings.filterwarnings("ignore")
import re, string
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ========== CONNECT DATASET ==========
DATA_PATH = "Flipkart Reviews Sentiment Analysis.csv"
# =====================================

for p in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.data.find("tokenizers/" + p if p.startswith("punkt") else "corpora/" + p)
    except LookupError:
        nltk.download(p, quiet=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}

# Phase 1 - Load & clean
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df = df.drop_duplicates(subset=["Review_ID"], keep="first")
df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
df["Year"] = df["Review_Date"].dt.year
df["Month"] = df["Review_Date"].dt.month
df["Day"] = df["Review_Date"].dt.day
df["DayOfWeek"] = df["Review_Date"].dt.day_name()
df["Review_Summary"] = df["Review_Summary"].fillna("")
df["Customer_City"] = df["Customer_City"].fillna("Unknown")
df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5) & (df["Delivery_Days"] >= 0)]
print("Cleaned:", df.shape)
print(df["Sentiment"].value_counts())

# Phase 2 - EDA plots
sc = df["Sentiment"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sc.plot(kind="bar", ax=ax[0], color=[colors.get(x, "gray") for x in sc.index])
ax[0].set_title("Sentiment counts")
ax[1].pie(sc, labels=sc.index, autopct="%1.1f%%", colors=[colors.get(x, "gray") for x in sc.index], startangle=90)
ax[1].set_title("Sentiment %")
plt.tight_layout()
plt.savefig("01_sentiment.png", dpi=120, bbox_inches="tight")
plt.close()

ct = pd.crosstab(df["Rating"], df["Sentiment"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ct.plot(kind="bar", stacked=True, ax=axes[0], color=[colors["Negative"], colors["Neutral"], colors["Positive"]])
axes[0].set_title("Rating vs Sentiment")
sns.boxplot(data=df, x="Sentiment", y="Rating", order=["Negative", "Neutral", "Positive"], palette=colors, ax=axes[1])
axes[1].set_title("Rating by Sentiment")
plt.tight_layout()
plt.savefig("02_rating_sentiment.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved EDA plots")

# Phase 3 - NLP
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def preprocess(t):
    toks = [x for x in word_tokenize(clean_text(t)) if x not in stop_words and len(x) > 2]
    return " ".join(lemmatizer.lemmatize(x) for x in toks)

df["Combined_Review"] = (df["Review_Summary"].fillna("") + " " + df["Review_Text"].fillna("")).str.strip()
df["Cleaned_Text"] = df["Combined_Review"].apply(preprocess)

# Phase 4 - ML
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
inv_map = {v: k for k, v in label_map.items()}
X = df["Cleaned_Text"].astype(str).values
y = np.array([label_map[s] for s in df["Sentiment"].values])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Linear SVM": LinearSVC(random_state=42, max_iter=2000),
}
results = {}
for name, clf in models.items():
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    results[name] = {
        "pred": pred,
        "acc": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="macro"),
    }
    print(f"{name}: Acc={results[name]['acc']:.4f} F1={results[name]['f1']:.4f}")

grid = GridSearchCV(
    LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
    {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
    cv=3, scoring="f1_macro", n_jobs=-1,
)
grid.fit(Xtr, y_train)
best_lr = grid.best_estimator_
pred_lr = best_lr.predict(Xte)
print("Tuned LR params:", grid.best_params_)
print("Tuned LR Acc:", round(accuracy_score(y_test, pred_lr), 4),
      "F1:", round(f1_score(y_test, pred_lr, average="macro"), 4))
print(classification_report(y_test, pred_lr, target_names=["Negative", "Neutral", "Positive"]))

# Phase 5 - DL (optional)
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, GlobalMaxPooling1D, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    MAX_VOCAB, MAX_LEN = 1500, 25
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(list(X_train))
    Xtr_pad = pad_sequences(tok.texts_to_sequences(list(X_train)), maxlen=MAX_LEN, padding="post").astype("int32")
    Xte_pad = pad_sequences(tok.texts_to_sequences(list(X_test)), maxlen=MAX_LEN, padding="post").astype("int32")
    ytr_cat = to_categorical(y_train, 3).astype("float32")
    rng = np.random.RandomState(42)
    n = min(8000, len(Xtr_pad))
    idx = rng.choice(len(Xtr_pad), n, replace=False)
    Xd, yd = Xtr_pad[idx], ytr_cat[idx]
    early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    bilstm = Sequential([
        Input(shape=(MAX_LEN,)), Embedding(MAX_VOCAB, 24),
        Bidirectional(LSTM(16)), Dense(16, activation="relu"), Dropout(0.2),
        Dense(3, activation="softmax"),
    ])
    bilstm.compile("adam", "categorical_crossentropy", ["accuracy"])
    bilstm.fit(Xd, yd, validation_split=0.15, epochs=5, batch_size=64, callbacks=[early], verbose=1)
    pred_dl = np.argmax(bilstm.predict(Xte_pad, verbose=0), axis=1)
    print("BiLSTM Acc:", round(accuracy_score(y_test, pred_dl), 4),
          "F1:", round(f1_score(y_test, pred_dl, average="macro"), 4))
except Exception as e:
    print("DL skipped:", e)

# Prediction pipeline
def predict_sentiment(review):
    cleaned = preprocess(review)
    vec = tfidf.transform([cleaned])
    idx = int(best_lr.predict(vec)[0])
    conf = float(best_lr.predict_proba(vec)[0][idx]) if hasattr(best_lr, "predict_proba") else None
    return inv_map[idx], conf

print("\n=== Demo predictions ===")
for r in [
    "Amazing product, works perfectly and quality is superb!",
    "Average quality, nothing special, okay for the price.",
    "Terrible experience, waste of money, defective product.",
]:
    s, c = predict_sentiment(r)
    print(f"[{s} {c:.3f}] {r}")
print("Done.")
