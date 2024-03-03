# Import necessary libraries
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from a CSV file
data = pd.read_csv("twitter.csv")

# Uncomment the line below to display the first few rows of the dataset
# print(data.head())

# Map numerical labels to descriptive categories
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})

# Uncomment the line below to display the updated dataset
# print(data.head())

# Select relevant columns for analysis
data = data[["tweet", "labels"]]

# Uncomment the line below to display the updated dataset
# print(data.head())

# Perform text cleaning on the tweet column
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply the cleaning function to the tweet column
data["tweet"] = data["tweet"].apply(clean)

# Uncomment the line below to display the cleaned dataset
# print(data.head())

# Split the dataset into features (X) and labels (y)
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Initialize the CountVectorizer to convert text data into a bag-of-words representation
cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the Decision Tree Classifier and fit it to the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
accuracy = clf.score(X_test, y_test)

# Uncomment the line below to display the accuracy of the model
# print(f"Model Accuracy: {accuracy}")

# Function to detect hate speech using the trained model
def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)
        st.title(prediction)

# Run the hate_speech_detection function when the script is executed
hate_speech_detection()
