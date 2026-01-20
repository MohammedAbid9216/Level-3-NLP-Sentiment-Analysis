# LEVEL 3 – TASK 2
# NLP Sentiment Analysis using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("sentiment_dataset.csv")
print("Dataset loaded successfully")
print(df.head())

# --------------------------------------------------
# 2. Keep only required columns
# --------------------------------------------------
df = df[['Text', 'Sentiment']]

# --------------------------------------------------
# 3. Clean Sentiment column
# --------------------------------------------------
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.capitalize()

# --------------------------------------------------
# 4. Convert Sentiment labels to numbers
# Positive -> 1, Neutral -> 0, Negative -> -1
# --------------------------------------------------
df['Sentiment'] = df['Sentiment'].map({
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
})

# --------------------------------------------------
# 5. Remove rows with NaN values
# --------------------------------------------------
df = df.dropna()

print("\nAfter cleaning Sentiment values:")
print(df['Sentiment'].value_counts())

# --------------------------------------------------
# 6. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['Sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['Sentiment']
)

# --------------------------------------------------
# 7. Text Vectorization (TF-IDF)
# --------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------------------------------
# 8. Train Logistic Regression Model
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --------------------------------------------------
# 9. Predictions & Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test_vec)

print("\nModel Evaluation Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
