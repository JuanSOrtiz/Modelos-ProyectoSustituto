from flask import Flask, request, jsonify
import subprocess
import sys
import pandas as pd

app = Flask(__name__)

MODEL_FILE = 'model.pkl'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
TEST_INPUT_FILE = 'test_data_input.csv'
TEST_TARGET_FILE = 'test_data_target.csv'
TEST_PREDS_FILE = 'test_predictions.csv'
PYTHON = sys.executable

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame(data)
        input_file = 'input_data.csv'
        input_df.to_csv(input_file, index=False)

        subprocess.run([PYTHON, 'predict.py', '--model_file', MODEL_FILE, '--input_file', input_file, '--predictions_file', TEST_PREDS_FILE], check=True)

        predictions = pd.read_csv(TEST_PREDS_FILE).values.flatten().tolist()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train', methods=['POST'])
def train():
    try:
        subprocess.run([PYTHON, 'train.py', '--model_file', MODEL_FILE, '--train_data_file', TRAIN_DATA_FILE, '--test_data_file', TEST_DATA_FILE, '--overwrite_model'], check=True)
        return jsonify({'message': 'Model training started'})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
