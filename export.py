import json
from pathlib import Path
from PIL import Image
import numpy as np
import nap
from nap import cache
import util

EXPORT_LOCATION = Path("activation_cluster_explorer/backend/data")


def export_labels(labels, model_name, destination=EXPORT_LOCATION):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    Path(destination, model_name).mkdir(parents=True, exist_ok=True)
    with open(Path(destination, model_name, "labels.json"), 'w') as outfile:
        json.dump(labels, outfile, cls=NpEncoder)


def export_patterns(model, model_name, X, layers, agg_func = np.mean, destination=EXPORT_LOCATION):
    for layer in layers:
        path = Path(destination, model_name, "layers", str(layer))
        path.mkdir(parents=True, exist_ok=True)
        patterns = nap.cache.get_layer_patterns(X, model, model_name, layer, agg_func)
        patterns.to_pickle(Path(path, "patterns.pkl"))


def export_images(model, model_name, X, layers, agg_func = np.mean, destination=EXPORT_LOCATION):
    import tensorflow as tf
    indices = set()
    for layer in layers:
        patterns = nap.cache.get_layer_patterns(X, model, model_name, layer, agg_func)
        sorted_patterns = nap.sort(patterns)
        for pattern_id, pattern in sorted_patterns.groupby('patternId'):
            if pattern_id == -1:
                continue
            indices.update(pattern.index)
            path = Path(destination, model_name, "layers", str(
                layer), str(int(pattern_id)))
            path.mkdir(parents=True, exist_ok=True)
            to_average = util.filter_tf_dataset(X, pattern.index)
            avg = tf.keras.layers.Average()(to_average).numpy()
            centers = pattern.head(1).index
            outliers = pattern.tail(3).index
            with open(Path(path, "centers.json"), 'w') as outfile:
                json.dump(centers.tolist(), outfile)
            with open(Path(path, "outliers.json"), 'w') as outfile:
                json.dump(outliers.tolist(), outfile)
            export_image(path, "average", avg)
    
    images_path = Path(destination, model_name, "images")
    images_path.mkdir(parents=True, exist_ok=True)
    to_export = util.filter_tf_dataset(X, list(indices))
    for index, image in zip(indices, to_export):
        export_image(images_path, f"{index}", image)

def export_image(path, name, array):
    image = np.squeeze(array)
    if (len(image.shape)):
        image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    else:
        image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    image.save(Path(path, f"{name}.jpeg"))


def export_all(model, model_name, X, y, layers, agg_func = np.mean, destination=EXPORT_LOCATION):
    export_labels(y, model_name, destination)
    export_patterns(model, model_name, X, layers, agg_func, destination)
    export_images(model, model_name, X, layers, agg_func, destination)
