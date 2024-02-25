import os
import pandas as pd

def load_data_from_folders(root_folder):
    categories = []
    texts = []

    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if filename.endswith(".txt") and os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            categories.append(category)
                            texts.append(text)
                    except UnicodeDecodeError:
                        print(f"Error reading file: {file_path}")

    df = pd.DataFrame({'category': categories, 'text': texts})
    return df

# Set the root folder containing category folders
root_folder = "JoshuaJ1080/Classification-of-News-Articls-using-ML/Dataset"
df = load_data_from_folders(root_folder)

"""Step 2 : Text Preprocessing"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Apply stemming
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())

    return text

# Apply text preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)

"""Step 3 :

Step 3.1 : Word Frequency Feature
"""

from sklearn.feature_extraction.text import CountVectorizer

# Create a count vectorizer
count_vectorizer = CountVectorizer(max_features=5000)  # Adjust max_features as needed

# Fit and transform the processed text
word_freq_feature = count_vectorizer.fit_transform(df['processed_text']).toarray()

"""Step 3.2 : Additional Features"""

# Document Length Feature
df['doc_length'] = df['processed_text'].apply(len)

# TF-IDF Feature
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_feature = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()

"""Step 4 : Feature Selection"""

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# Combine all features
all_features = np.concatenate((word_freq_feature, df[['doc_length']].values, tfidf_feature), axis=1)

# Use chi-squared test for feature selection
k_best_features = SelectKBest(chi2, k=1000).fit_transform(all_features, df['category'])

"""Step 5 : Train-Test Split"""

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(k_best_features, df['category'], test_size=0.2, random_state=42)

"""Step 6 : Model Training"""

from sklearn.ensemble import RandomForestClassifier

# Example: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""Step 7 : Model Evaluation"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


"""Overall Performance"""

from sklearn.metrics import precision_recall_fscore_support

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score for each category
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Macro-Averaged Precision: {precision}")
print(f"Macro-Averaged Recall: {recall}")
print(f"Macro-Averaged F1 Score: {f1}")
