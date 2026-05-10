from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle, os
import warnings
import subprocess

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

BASE   = os.path.dirname(os.path.abspath(__file__))
model  = tf.keras.models.load_model(os.path.join(BASE, 'best_model.h5'))
scaler = pickle.load(open(os.path.join(BASE, 'scaler.pkl'), 'rb'))

print("✓ Modèle chargé:", model.input_shape)
print("✓ Scaler chargé:", scaler.n_features_in_, "features")
print("✓ API démarrée sur http://0.0.0.0:5005")

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': 'CNN1D - 94.69% accuracy',
        'features': 42,
        'threshold': 0.5
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédit une seule session"""
    try:
        features = request.json.get('features', [])
        if len(features) != 42:
            return jsonify({'error': f'42 features required, got {len(features)}'}), 400
        
        X     = np.array(features).reshape(1, -1)
        X_sc  = scaler.transform(X)
        X_seq = X_sc.reshape(1, 1, X_sc.shape[1])
        prob  = float(model.predict(X_seq, verbose=0)[0][0])
        
        return jsonify({
            'score'     : round(prob * 100, 2),
            'label'     : 'ATTAQUE' if prob > 0.5 else 'NORMAL',  # THRESHOLD = 0.5
            'confidence': round(max(prob, 1-prob) * 100, 2),
            'raw_prob'  : round(prob, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Prédit plusieurs sessions en batch"""
    try:
        sessions = request.json.get('sessions', [])
        results  = []
        
        for s in sessions:
            features = s.get('features', [])
            if not features or len(features) != 42:
                continue
            
            X     = np.array(features).reshape(1, -1)
            X_sc  = scaler.transform(X)
            X_seq = X_sc.reshape(1, 1, X_sc.shape[1])
            prob  = float(model.predict(X_seq, verbose=0)[0][0])
            
            results.append({
                'ip'   : s.get('src_ip', 'unknown'),
                'score': round(prob * 100, 2),
                'label': 'ATTAQUE' if prob > 0.5 else 'NORMAL',  # THRESHOLD = 0.5
                'confidence': round(max(prob, 1-prob) * 100, 2)
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)


@app.route("/block_ip", methods=["POST"])
def block_ip():

    data = request.json
    ip = data.get("ip")

    if not ip:
        return jsonify({
            "status": "error",
            "message": "missing ip"
        })

    try:

        # Linux iptables
        subprocess.run([
            "sudo",
            "iptables",
            "-A",
            "INPUT",
            "-s",
            ip,
            "-j",
            "DROP"
        ])

        return jsonify({
            "status": "success",
            "ip": ip
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)