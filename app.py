from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle, os

app = Flask(__name__)
CORS(app)

BASE   = os.path.dirname(os.path.abspath(__file__))
model  = tf.keras.models.load_model(os.path.join(BASE, 'best_model.h5'))
scaler = pickle.load(open(os.path.join(BASE, 'scaler.pkl'), 'rb'))
print("Modele charge :", model.input_shape)
print("Scaler charge :", scaler.n_features_in_, "features")

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'CNN1D'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json.get('features', [])
        X     = np.array(features).reshape(1, -1)
        X_sc  = scaler.transform(X)
        X_seq = X_sc.reshape(1, 1, X_sc.shape[1])
        prob  = float(model.predict(X_seq, verbose=0)[0][0])
        return jsonify({
            'score'     : round(prob * 100, 2),
            'label'     : 'ATTAQUE' if prob > 0.5 else 'NORMAL',
            'confidence': round(max(prob, 1-prob) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        sessions = request.json.get('sessions', [])
        results  = []
        for s in sessions:
            features = s.get('features', [])
            if not features:
                continue
            X     = np.array(features).reshape(1, -1)
            X_sc  = scaler.transform(X)
            X_seq = X_sc.reshape(1, 1, X_sc.shape[1])
            prob  = float(model.predict(X_seq, verbose=0)[0][0])
            results.append({
                'ip'   : s.get('src_ip', ''),
                'score': round(prob * 100, 2),
                'label': 'ATTAQUE' if prob > 0.5 else 'NORMAL'
            })
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False)