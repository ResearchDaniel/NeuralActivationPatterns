import json
from pathlib import Path
from PIL import Image
import numpy as np
import nap

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


def export_patterns(ap, X, model_name, layers, destination=EXPORT_LOCATION):
    for layer in layers:
        path = Path(destination, model_name, "layers", str(layer))
        path.mkdir(parents=True, exist_ok=True)
        ap.layer_patterns(layer, X).to_pickle(Path(path, "patterns.pkl"))


def export_images(ap, X, model_name, layers, destination=EXPORT_LOCATION):
    images_path = Path(destination, model_name, "images")
    images_path.mkdir(parents=True, exist_ok=True)
    images = list(X.unbatch().as_numpy_iterator())
    for index, image in enumerate(images):
        export_image(images_path, f"{index}", image)
    for layer in layers:
        patterns = ap.layer_patterns(layer, X)
        sorted_patterns = nap.sort(patterns)
        for pattern_id, pattern in sorted_patterns.groupby('patternId'):
            path = Path(destination, model_name, "layers", str(
                layer), str(int(pattern_id)))
            path.mkdir(parents=True, exist_ok=True)
            avg = nap.average(images, pattern.index)
            centers = pattern.head(1).index
            outliers = pattern.tail(3).index
            with open(Path(path, "centers.json"), 'w') as outfile:
                json.dump(centers.tolist(), outfile)
            with open(Path(path, "outliers.json"), 'w') as outfile:
                json.dump(outliers.tolist(), outfile)
            export_image(path, "average", avg)


def export_image(path, name, array):
    image = np.squeeze(array)
    if (len(image.shape)):
        image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    else:
        image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    image.save(Path(path, f"{name}.jpeg"))


def export_all(ap, X, y, name, layers, destination=EXPORT_LOCATION):
    export_labels(y, name, destination)
    export_patterns(ap, X, name, layers, destination)
    export_images(ap, X, name, layers, destination)
