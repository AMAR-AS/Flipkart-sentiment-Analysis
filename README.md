# Flipkart Reviews Sentiment Analysis

End-to-end **NLP + Machine Learning + Deep Learning** student project for 3-class sentiment classification (Positive / Neutral / Negative) on Flipkart-style product reviews.

## Dataset

| Item | Value |
|------|-------|
| Size | 30,000 reviews × 25 columns |
| Target | Sentiment (Positive / Neutral / Negative) |
| Class balance | ~62% Positive · ~20% Neutral · ~18% Negative |
| Primary metric | **Macro F1-Score** |

Place your CSV as `Flipkart Reviews Sentiment Analysis.csv` next to the notebook (or set `DATA_PATH` inside the notebook).

## Project structure

```
Flipkart-sentiment-Analysis/
├── README.md
├── requirements.txt
├── Flipkart_Sentiment_Standalone.ipynb   # Main runnable notebook (self-contained)
├── notebooks/
│   └── Flipkart_Sentiment_Analysis_Explain.ipynb
└── reports/
    ├── Final_Project_Report.md
    └── model_comparison.csv
```

## Quick start

```bash
pip install -r requirements.txt
# Download NLTK data once (or let the notebook auto-download)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
```

Open **`Flipkart_Sentiment_Standalone.ipynb`**, set:

```python
DATA_PATH = "Flipkart Reviews Sentiment Analysis.csv"
```

Run all cells top to bottom.

## Pipeline (50 tasks)

1. **Data prep** – load, missing values, duplicates, dates, cleaning  
2. **EDA** – sentiment distribution, rating vs sentiment, categories, brands, word frequency plots  
3. **NLP** – combine text → clean → tokenize → stop-words → lemmatize  
4. **Traditional ML** – TF-IDF (uni+bi) → Naive Bayes, Logistic Regression, Linear SVM → **Tuned LR** (GridSearchCV)  
5. **Deep Learning** – Embedding + Dense, LSTM, BiLSTM  
6. **Prediction pipeline** – `predict_sentiment(review)` → class + confidence  

## Model results (this synthetic dataset)

| Model | Type | Accuracy | Macro F1 |
|-------|------|----------|----------|
| **Tuned Logistic Regression** | Traditional ML | 1.00 | 1.00 |
| **BiLSTM** | Deep Learning | 1.00 | 1.00 |
| Dense NN | Deep Learning | ~1.00 | ~1.00 |
| LSTM | Deep Learning | ~0.97 | ~0.94 |

Scores are near-perfect because reviews are short and template-like. On real Flipkart text, expect lower scores; the same pipeline still applies.

**Recommended production model here:** Tuned Logistic Regression (same quality as BiLSTM, faster, interpretable).

### Tuned LR (what it is)

- Multinomial logistic regression on TF-IDF features  
- `class_weight='balanced'` for class imbalance  
- GridSearch over `C` and `solver` maximizing **macro F1**  
- Coefficients show top positive/negative words  

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn, nltk  
- tensorflow (for DL cells; optional if you only run ML)

## Author

AMAR-AS — B.Tech student project (ML & Deep Learning)
