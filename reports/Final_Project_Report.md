# Flipkart Reviews Sentiment Analysis – Final Project Report

**Student Project – Machine Learning & Deep Learning**  
**Dataset:** 30,000 synthetic Flipkart-style reviews × 25 columns  
**Primary Target:** Sentiment (Positive / Neutral / Negative)  
**Primary Metric:** Macro F1-Score  

---

## 1. Problem Statement

E-commerce platforms receive large volumes of customer reviews. Manually reading every review is impractical. An automated sentiment analysis system classifies feedback as Positive, Neutral, or Negative, helping organizations understand product perception, customer experience, service quality, and emerging issues.

## 2. Dataset Overview

| Item | Value |
|------|-------|
| Source file | Flipkart Reviews Sentiment Analysis.csv |
| Records | 30,000 |
| Columns | 25 |
| Target | Sentiment (3 classes) |
| Class distribution | Positive 18,671 (62.2%) · Neutral 6,019 (20.1%) · Negative 5,310 (17.7%) |

**Key columns used for NLP:** Review_Summary + Review_Text → Combined_Review → Cleaned_Text (lemmatized).

**Data quality notes:**
- Missing values: Review_Summary (240), Customer_City (300), Payment_Method (150) – filled appropriately.
- No duplicate Review_IDs.
- Rating, prices, delivery days within realistic ranges after minor cleaning.
- Review text is short and highly template-driven (educational synthetic data), leading to strong linear separability.

## 3. Methodology

### Phase 1 – Data Preparation
- Loaded CSV, inspected shape/dtypes/missing values.
- Converted Review_Date → Year, Month, Day, DayOfWeek.
- Handled missing values, removed invalid records, saved cleaned dataset.

### Phase 2 – Exploratory Data Analysis
- Sentiment distribution (bar + pie).
- Rating vs Sentiment (stacked bar + boxplot).
- Category & brand review volume rankings.
- Average rating and selling price by category.
- Discount percentage vs sentiment.
- Delivery days vs Customer_Experience_Score (weak correlation).
- Verified vs non-verified purchase sentiment (similar distributions).
- Word-frequency analysis per sentiment class.

### Phase 3 – NLP Preprocessing
1. Combined Review_Summary + Review_Text.
2. Lowercasing + removal of punctuation/numbers/special characters.
3. Tokenization (NLTK).
4. English stop-word removal (avg word count reduced).
5. Stemming (Porter) vs Lemmatization (WordNet) comparison on 20 samples.
6. Final representation: **Lemmatized Cleaned_Text** (preferred for semantic fidelity).

Average cleaned length: ~7–8 tokens per review.

### Phase 4 – Traditional Machine Learning
- **Features:** TF-IDF (unigram + bigram, max_features=5000, min_df=5).
- **Split:** Stratified 80/20.
- **Models:** Multinomial NB, Logistic Regression, Linear SVM.
- **Best model:** Tuned Logistic Regression (class_weight=balanced + GridSearchCV on C/solver).
- Macro F1 ≈ 1.00 on this synthetic set.
- Top positive features: excellent, superb, loved, worth, amazing…
- Top negative features: waste, defective, poor, disappointed, bad…

### Phase 5 – Deep Learning
- Keras Tokenizer + padding (maxlen=25).
- Models: Embedding + Dense, LSTM, BiLSTM.
- **Best DL:** BiLSTM (Macro F1 ≈ 1.00).

### Prediction Pipeline
Reusable function: preprocess → TF-IDF → Tuned LR → class + confidence.

## 4. Results Summary

| Model | Type | Accuracy | Macro F1 |
|-------|------|----------|----------|
| Tuned Logistic Regression | Traditional ML | 1.0000 | 1.0000 |
| BiLSTM | Deep Learning | 1.0000 | 1.0000 |
| Dense NN | Deep Learning | 0.9998 | 0.9998 |
| LSTM | Deep Learning | 0.9673 | 0.9424 |

**Recommendation:** Tuned Logistic Regression for production (same performance, faster, interpretable).

## 5. All 50 Tasks

Phases 1–5 fully covered (data prep, EDA, NLP, ML, DL, prediction pipeline).

## 6. Future Scope

Word2Vec/FastText/BERT, attention, Streamlit app, SHAP/LIME, real Flipkart data, temporal trends.

## 7. Conclusion

End-to-end NLP + ML + DL pipeline completed on 30k Flipkart-style reviews. Portfolio-ready project from raw CSV to deployable classifier.
