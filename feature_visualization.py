"""Compute layer/unit feature visualizations for a model."""
from pathlib import Path
import tensorflow as tf
from luna.featurevis import featurevis, images
from luna.featurevis.transformations import standard_transformation


def layer_feature_visualizations(model, output_dir, layers):
    """Computes layer-level feature visualization (deep dream) and
    stores it in output_dir/layer_name/layer_feature_vis.png

    Args:
        model (object): Keras model to compute feature visualizations for.
        output_dir (Path): Model directory to store the results in.
        layers (List[str]): List of layer names to compute feature vizualizations.
    """
    opt_param = featurevis.OptimizationParameters(
        iterations=200, learning_rate=0.05, optimizer=tf.keras.optimizers.Adam(epsilon=1e-08))
    _, width, height, channels = model.get_config()["layers"][0]["config"]["batch_input_shape"]
    for layer in layers:
        layer_dir = Path(output_dir, layer)
        layer_dir.mkdir(parents=True, exist_ok=True)
        feature_vis_path = Path(layer_dir, "layer_feature_vis.png")
        if feature_vis_path.exists():
            continue

        opt_image = images.initialize_image_ref(
            width, height, channels=channels, fft=True, decorrelate=True, seed=1)
        layer_objective = featurevis.objectives.LayerObjective(model, layer=layer)
        _, deep_dream = featurevis.visualize(
            opt_image, objective=layer_objective, optimization_parameters=opt_param,
            transformation=standard_transformation)
        deep_dream_img = tf.keras.preprocessing.image.array_to_img(deep_dream)
        deep_dream_img.save(feature_vis_path)


def filter_feature_visualizations(model, output_dir, layers, filters,
                                  opt_param=featurevis.OptimizationParameters(
                                      iterations=200, learning_rate=0.05,
                                      optimizer=tf.keras.optimizers.Adam(epsilon=1e-08))):
    """Computes filter-level feature visualization and
    stores it in output_dir/layer_name/filter_feature_vis/filter_index.png
    Args:
        model (object): Keras model to compute feature visualizations for.
        output_dir (Path): Model directory to store the results in.
        layers (List[str]): List of layer names to compute feature vizualizations.
        filters (object): Dict of filter indices to compute {layer_name:[filters]}.
    """
    shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
    for layer in layers:
        feature_vis_dir = Path(output_dir, layer, "filter_feature_vis")
        if layer in filters:
            for filter_index in filters[layer]:
                feature_vis_path = Path(feature_vis_dir, f"{filter_index}.png")
                if feature_vis_path.exists():
                    continue
                opt_image = images.initialize_image_ref(
                    shape[1], shape[2], channels=shape[3], fft=True, decorrelate=True, seed=1)
                filter_objective = featurevis.objectives.FilterObjective(
                    model, layer=layer, filter_index=filter_index)
                try:
                    _, filter_vis = featurevis.visualize(
                        opt_image, objective=filter_objective, optimization_parameters=opt_param,
                        transformation=standard_transformation)
                    # Do not create directory until we know that feature visualization is succesful.
                    feature_vis_dir.mkdir(parents=True, exist_ok=True)
                    tf.keras.preprocessing.image.array_to_img(filter_vis).save(feature_vis_path)
                except tf.errors.InvalidArgumentError:
                    continue
