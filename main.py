import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ===== Step 1: Load and Prepare Data =====
print("[INFO] Loading dataset...")
data = pd.read_csv("fake_or_real_news.csv")

# Encode labels: REAL → 0, FAKE → 1
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)

X = data['text']
y = data['fake']

# ===== Step 2: Split Data =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Step 3: TF-IDF Vectorization =====
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== Step 4: Train Classifier =====
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# ===== Step 5: Evaluate Model =====
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Accuracy: {acc:.4f}")
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

# ===== Step 6: Test on a Random Example =====
index = 100
sample_text = X_test.iloc[index]
sample_label = y_test.iloc[index]

vectorized_sample = vectorizer.transform([sample_text])
prediction = clf.predict(vectorized_sample)[0]

print("\n=== Sample Test ===")
print("Text:", sample_text[:500], "...\n")
print("Actual Label:", "REAL" if sample_label == 0 else "FAKE")
print("Predicted Label:", "REAL" if prediction == 0 else "FAKE")

# ===== Step 7: Optional User Input =====
def predict_news(text):
    vect = vectorizer.transform([text])
    pred = clf.predict(vect)[0]
    return "REAL" if pred == 0 else "FAKE"

# Example
# user_input = input("\nEnter a news headline or paragraph:\n> ")
# print("Prediction:", predict_news(user_input))
