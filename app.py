from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load trained model and scaler
with open("kmeans_model.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        scaled_features = scaler.transform(features)
        cluster = kmeans.predict(scaled_features)[0]
        return jsonify({'Cluster': int(cluster)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
