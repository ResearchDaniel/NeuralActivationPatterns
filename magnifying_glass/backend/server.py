"""Server interface for the latent retrieval demo."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from flask import Flask, jsonify, request, send_file


class NpEncoder(json.JSONEncoder):
    """An encoder that can handle numpy data types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


app = Flask(__name__)
app.json_provider_class = NpEncoder
OK_STATUS = 200
ERROR_STATUS = 400
TEXT_TYPE = {"ContentType": "text/plain"}
JSON_TYPE = {"ContentType": "application/json"}
DATA_DIR = Path("data")


@app.route("/api/get_models")
def get_models():
    nets = []
    for path in DATA_DIR.iterdir():
        if path.is_dir():
            nets.append(path.name)
    networks = {"networks": nets}
    return jsonify(networks)


@app.route("/api/get_layers/<model>")
def get_layers(model):
    layer_dirs = []
    for path in Path(DATA_DIR, model, "layers").iterdir():
        if path.is_dir():
            layer_dirs.append(path.name)
    with open(Path(DATA_DIR, model, "config.json"), encoding="utf8") as json_file:
        data = json.load(json_file)
    if "layers" in data:
        layers = list(filter(lambda layer: layer in layer_dirs, data["layers"]))
        layers = {"layers": layers}
    else:
        layers = {"layers": layer_dirs}
    return jsonify(layers)


@app.route("/api/get_pattern_statistics/<model>/<layer>/<pattern>")
def get_pattern_statistics(model, layer, pattern):
    arrow_path = Path(
        DATA_DIR, model, "layers", layer, f"pattern_statistics_{pattern}.arrow"
    )
    if not arrow_path.exists():
        return "No such file", ERROR_STATUS
    return send_file(arrow_path, mimetype="application/octet-stream")


@app.route("/api/get_patterns/<model>/<layer>")
def get_patterns(model, layer):
    pickle_path = Path(DATA_DIR, model, "layers", layer, "patterns.pkl")
    if not pickle_path.exists():
        return "No such file", ERROR_STATUS
    patterns = pd.read_pickle(pickle_path)
    return jsonify(patterns.to_json(orient="records"))


@app.route("/api/get_max_activations/<model>/<layer>")
def get_max_activating(model, layer):
    max_path = Path(DATA_DIR, model, "layers", layer, "max_activations.json")
    if not max_path.exists():
        return jsonify({"max_activations": []})
    return send_file(max_path, mimetype="application/json")


@app.route("/api/get_layer_feature_vis_exists/<model>/<layer>")
def get_layer_feature_vis_exists(model, layer):
    vis_path = Path(DATA_DIR, model, "layers", layer, "layer_feature_vis.png")
    if not vis_path.exists():
        return jsonify(False)
    return jsonify(True)


@app.route("/api/get_layer_feature_vis/<model>/<layer>")
def get_layer_feature_vis(model, layer):
    return send_file(
        Path(DATA_DIR, model, "layers", layer, "layer_feature_vis.png"),
        mimetype="image/png",
    )


@app.route("/api/get_pattern_info/<model>/<layer>")
def get_pattern_info(model, layer):
    pickle_path = Path(DATA_DIR, model, "layers", layer, "patterns_info.pkl")
    if not pickle_path.exists():
        return "No such file", ERROR_STATUS
    patterns = pd.read_pickle(pickle_path)
    return jsonify(patterns.to_json(orient="records"))


@app.route("/api/get_dataset/<model>")
def get_dataset(model):
    dataset = pd.read_pickle(Path(DATA_DIR, model, "dataset.pkl"))
    return jsonify(dataset.to_json(orient="records"))


@app.route("/api/get_average/<model>/<layer>/<pattern>")
def get_average(model, layer, pattern):
    return send_file(
        Path(DATA_DIR, model, "layers", layer, pattern, "average.jpeg"),
        mimetype="image/jpeg",
    )


@app.route("/api/get_image/<model>/<identifier>")
def get_image(model, identifier):
    with open(Path(DATA_DIR, model, "config.json"), encoding="utf8") as json_file:
        data = json.load(json_file)
    return send_file(Path(data["data_path"], identifier), mimetype="image/jpeg")


@app.route("/api/get_labels/<model>")
def get_labels(model):
    with open(Path(DATA_DIR, model, "config.json"), encoding="utf8") as json_file:
        data = json.load(json_file)
    labels = pd.read_pickle(Path(data["data_path"], "label_names.pkl"))
    return jsonify(labels)


@app.route("/api/get_image_patterns", methods=["POST"])
def get_image_patterns():
    pattern_result = []
    request_data = json.loads(request.data)
    for model in set(map(lambda x: x["model"], request_data)):
        images = map(
            lambda x: x["image"],
            filter(
                lambda item, current_model=model: item["model"] == current_model,
                request_data,
            ),
        )
        dataset = pd.read_pickle(Path(DATA_DIR, model, "dataset.pkl"))
        image_rows = dataset.loc[dataset["file_name"].isin(images)]
        if not image_rows.empty:
            for layer in json.loads(get_layers(model).data)["layers"]:
                patterns_path = Path(DATA_DIR, model, "layers", layer, "patterns.pkl")
                if patterns_path.exists():
                    patterns = pd.read_pickle(patterns_path)
                    pattern_indices = set(
                        filter(
                            lambda pattern: pattern != -1,
                            patterns.loc[image_rows.index.values]["patternId"].tolist(),
                        )
                    )
                    stats_path = Path(
                        DATA_DIR, model, "layers", layer, "patterns_statistics.pkl"
                    )
                    statistics = None
                    if stats_path.exists():
                        statistics = pd.read_pickle(stats_path)
                    pattern_info = pd.read_pickle(
                        Path(DATA_DIR, model, "layers", layer, "patterns_info.pkl")
                    )
                    for pattern_index in pattern_indices:
                        pattern_samples = patterns.loc[
                            patterns["patternId"] == pattern_index
                        ].join(dataset)
                        pattern_result.append(
                            assemble_pattern(
                                statistics,
                                pattern_samples,
                                pattern_index,
                                pattern_info,
                                model,
                                layer,
                            )
                        )
    return jsonify(pattern_result), OK_STATUS


def assemble_pattern(
    statistics, pattern_samples, pattern_index, pattern_info, model, layer
):
    if statistics is None:
        return {
            "samples": pattern_samples.to_json(orient="records"),
            "persistence": pattern_info.loc[pattern_index]["pattern_persistence"],
            "model": model,
            "layer": layer,
            "labels": json.loads(get_labels(model).data),
        }
    return {
        "samples": pattern_samples.to_json(orient="records"),
        "statistics": json.dumps(statistics[pattern_index], cls=NpEncoder),
        "persistence": pattern_info.loc[pattern_index]["pattern_persistence"],
        "model": model,
        "layer": layer,
        "labels": json.loads(get_labels(model).data),
    }


if __name__ == "__main__":
    app.run(debug=True)
