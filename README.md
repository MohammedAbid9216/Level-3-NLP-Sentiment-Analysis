# Level 3 – NLP Sentiment Analysis

## 📌 Overview
This project is part of the **Codveda Technology Internship – Level 3 (Advanced)**.
The task focuses on **Natural Language Processing (NLP)** and aims to build a
machine learning model that can classify text into different sentiment categories.

---

## 🧠 Task Description
**Task Name:** NLP – Text Classification  
**Level:** Level 3 (Advanced)

The objective of this task is to:
- Preprocess text data
- Convert text into numerical features
- Train a classification model
- Evaluate the model using standard metrics

---

## 📂 Dataset
**File:** `sentiment_dataset.csv`

The dataset contains:
- Text data (social media posts)
- Sentiment labels: Positive, Neutral, Negative
- Additional metadata (timestamp, country, platform, etc.)

Only **Text** and **Sentiment** columns are used for modeling.

---

## ⚙️ Tools & Technologies
- Python
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 🔄 Workflow
1. Load and inspect the dataset
2. Clean and standardize sentiment labels
3. Convert sentiment labels into numerical form
4. Apply TF-IDF for text vectorization
5. Split data into training and testing sets
6. Train a Logistic Regression classifier
7. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

---

## ▶️ How to Run
1. Open terminal in the project directory
2. Run the command:
```bash
py level3_task2_nlp.py
