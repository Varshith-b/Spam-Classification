from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
classifier = pickle.load(open('final_model.pkl', 'rb'))
tfidf = pickle.load(open('final_vector.pkl', 'rb'))

# Define the prediction function
def predict_message(message):
    message_transformed = tfidf.transform([message]).toarray()
    prediction = classifier.predict(message_transformed)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        result = predict_message(message)
        return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
