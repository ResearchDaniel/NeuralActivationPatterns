"""Server interface for the latent retrieval demo."""
import json
from pathlib import Path
import pandas as pd

from flask import Flask, jsonify, send_file

app = Flask(__name__)
OK_STATUS = 200
ERROR_STATUS = 400
TEXT_TYPE = {'ContentType': 'text/plain'}
JSON_TYPE = {'ContentType': 'application/json'}
DATA_DIR = Path('data')


@app.route('/api/get_models')
def get_models():
    nets = []
    for path in DATA_DIR.iterdir():
        if path.is_dir():
            nets.append(path.stem)
    networks = {
        'networks': nets
    }
    return jsonify(networks)


@app.route('/api/get_layers/<model>')
def get_layers(model):
    layers = []
    for path in Path(DATA_DIR, model, "layers").iterdir():
        if path.is_dir():
            layers.append(path.stem)
    layers = {
        'layers': layers
    }
    return jsonify(layers)


@app.route('/api/get_patterns/<model>/<layer>')
def get_patterns(model, layer):
    patterns = pd.read_pickle(
        Path(DATA_DIR, model, "layers", layer, "patterns.pkl"))
    return jsonify(patterns.to_json(orient="records"))


@app.route('/api/get_labels/<model>')
def get_labels(model):
    with open(Path(DATA_DIR, model, 'labels.json')) as json_file:
        data = json.load(json_file)
    return jsonify(data)


@app.route('/api/get_average/<model>/<layer>/<pattern>')
def get_average(model, layer, pattern):
    return send_file(Path(DATA_DIR, model, "layers", layer, pattern, 'average.jpeg'), mimetype='image/jpeg')


@app.route('/api/get_image/<model>/<id>')
def get_image(model, id):
    with open(Path(DATA_DIR, model, 'config.json')) as json_file:
        data = json.load(json_file)
    return send_file(Path(data["data_path"], id), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
