from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and column data
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        print("Received Input Data:", data)  # Debugging

        total_sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        location = data['location'].strip().lower()

        # Ensure location exists in feature list
        if location not in data_columns:
            return jsonify({'estimated_price': "Invalid Location"})

        # Prepare input array
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        loc_index = data_columns.index(location)
        x[loc_index] = 1  # One-hot encode location

        # Predict price
        prediction = model.predict([x])[0]
        estimated_price = round(prediction, 2)

        print("Predicted Price (Before Adjustments):", estimated_price)

        # Ensure no zero prices
        estimated_price = max(10, estimated_price)  # Set min price to 10 Lakhs

        print("Final Estimated Price:", estimated_price)
        return jsonify({'estimated_price': estimated_price})

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': str(e)})

@app.route('/get_locations', methods=['GET'])
def get_locations():
    return jsonify({'locations': data_columns[3:]})  # Skip first three columns (sqft, bath, bhk)

if __name__ == '__main__':
    app.run(debug=True)
