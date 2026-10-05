# Flipkart Reviews Sentiment Analysis

End-to-end **NLP + Machine Learning + Deep Learning** project for 3-class sentiment (Positive / Neutral / Negative) on Flipkart-style reviews.

**Repo:** [AMAR-AS/Flipkart-sentiment-Analysis](https://github.com/AMAR-AS/Flipkart-sentiment-Analysis)

## Dataset

| Item | Value |
|------|-------|
| Size | 30,000 reviews × 25 columns |
| Target | Sentiment (Positive / Neutral / Negative) |
| Balance | ~62% Positive · ~20% Neutral · ~18% Negative |
| Metric | **Macro F1-Score** |

Put your CSV next to the scripts, or set `DATA_PATH` inside the file.

## Quick start

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"

# Edit DATA_PATH in the script if needed, then:
python flipkart_sentiment_pipeline.py
```

Or open **`Flipkart_Sentiment_Standalone.ipynb`** and Run All (set `DATA_PATH` in the first code cell).

## Pipeline

1. **Data prep** – load, clean, dates, missing values  
2. **EDA** – sentiment charts, rating vs sentiment, categories  
3. **NLP** – combine → clean → tokenize → stop-words → **lemmatize**  
4. **ML** – TF-IDF → NB / LR / SVM → **Tuned Logistic Regression** (GridSearchCV)  
5. **DL** – BiLSTM (optional if TensorFlow installed)  
6. **Predict** – `predict_sentiment(review)` → class + confidence  

## Results (synthetic dataset)

| Model | Type | Accuracy | Macro F1 |
|-------|------|----------|----------|
| **Tuned Logistic Regression** | ML | 1.00 | 1.00 |
| **BiLSTM** | DL | 1.00 | 1.00 |
| Dense / LSTM | DL | ~0.97–1.00 | ~0.94–1.00 |

Near-perfect scores reflect short, template-like synthetic reviews. Prefer **Tuned LR** for deployment (fast + interpretable).

### Tuned LR

- Multinomial logistic regression on TF-IDF (uni+bi)
- `class_weight='balanced'`
- GridSearch on `C` and `solver` maximizing macro F1

## Files

```
├── README.md
├── requirements.txt
├── flipkart_sentiment_pipeline.py   # full script
├── Flipkart_Sentiment_Standalone.ipynb
└── reports/
    ├── Final_Project_Report.md
    └── model_comparison.csv
```

## Author

**AMAR-AS** — B.Tech student project (ML & Deep Learning)
