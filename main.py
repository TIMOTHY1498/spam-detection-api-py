import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from fastapi import FastAPI
from pydantic import BaseModel

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv' # List message that used for the AI Training, please use your own!
df = pd.read_table(url, header=None, names=['label', 'message'])

# Text preprocessing
stop_words = set(stopwords.words('english'))
df['message'] = df['message'].str.lower().str.split()
df['message'] = df['message'].apply(lambda x: [word for word in x if word not in stop_words])
df['message'] = df['message'].str.join(' ')

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Test the model with a new message
def predict_spam(message):
    message = ' '.join([word for word in message.lower().split() if word not in stop_words])
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return prediction[0]

# Example usage

app = FastAPI()

# Define a Pydantic model for the request body
class Item(BaseModel):
    message: str


@app.post("/")
async def create_item(item: Item):
    if predict_spam(item.message) == "spam":
        return {"isSpam": True}
    else:
        return {"isSpam": False}
