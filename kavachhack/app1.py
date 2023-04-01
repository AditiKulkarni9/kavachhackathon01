import pickle
from flask import Flask, request, jsonify,render_template
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return render_template("front.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input string and image file from the form data
    input_string = request.form['inputString']
    #input_image = request.files['inputImage']
    cv= CountVectorizer()
    df=cv.transform([input_string]).toarray()
    prediction=model.predict(df)
    print(input_string)
    print(prediction)
    
    # Process the input data as needed (e.g., convert the image file to a numerical array)
    # ...
    
    # Make a prediction using the loaded model
    
    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
