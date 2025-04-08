import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"D:\Files\All my project folder\dvc_project_11\dvc_project_11\data\hugging_face.csv")  # Replace with your dataset file path
X = df['text']  # Assuming 'message' column contains text
y = df['label']    # Assuming 'label' column contains spam (1) or ham (0)

# Preprocess and vectorize text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

# Train the model
model = SVC(kernel='linear') 
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model and vectorizer
joblib.dump(model, "spam_ham_model_1.pkl")
joblib.dump(vectorizer, "vectorizer_1.pkl")