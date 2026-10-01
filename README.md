# Product Review Sentiment Analysis Using Transformer Embeddings

## Objective

* Built a **three-class sentiment classification system** for product reviews.
* Used the pre-trained **`all-MiniLM-L6-v2` Sentence Transformer** to convert review text into **384-dimensional embeddings**.
* Trained **Random Forest** and **Gradient Boosting** classifiers on the generated embeddings.
* Classified reviews into **Positive, Neutral, and Negative** sentiments.

## Dataset

* **1,005 processed product reviews**
* Features:

  * `Product ID`
  * `Product Review`
  * `Sentiment`
* The dataset contains **imbalanced sentiment classes**.
* Performed **duplicate and missing-value checks** during preprocessing.

## Tech Stack

* **Python**
* **Sentence Transformers**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

## Approach

```text
Product Reviews
       ↓
Data Cleaning
       ↓
Sentence Transformer
all-MiniLM-L6-v2
       ↓
384-Dimensional Embeddings
       ↓
80/20 Stratified Split
       ↓
Random Forest / Gradient Boosting
       ↓
Sentiment Prediction
```

## Models & Results

| Model             | Test Accuracy | Weighted F1 |
| ----------------- | ------------: | ----------: |
| Random Forest     |    **86.57%** |  **81.80%** |
| Gradient Boosting |        84.08% |      80.33% |

### Final Model

**Random Forest with Transformer Embeddings**

* **Test Accuracy:** 86.57%
* **Weighted F1 Score:** 81.80%

## Key Techniques

* Natural Language Processing
* Transformer-based Sentence Embeddings
* Text Classification
* Exploratory Data Analysis
* Stratified Train-Test Split
* Random Forest
* Gradient Boosting
* Accuracy and Weighted F1 Evaluation
* Imbalanced Dataset Evaluation

## Project Structure

```text
Product-Review-Sentiment-Analysis/
│
├── Case_Study_Product_Review_Sentiment_Analysis_Transformers.ipynb
├── Product_Reviews.csv
└── README.md
```

## Future Improvements

* Fine-tune a Transformer model for sentiment classification.
* Perform hyperparameter tuning to improve model performance.
* Deploy the sentiment analysis model using **Streamlit** or an API.
