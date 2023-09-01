"""TODO model package docstring."""
from typing import List, Tuple
import logging
import onnxruntime
import json
import os
import base64
from scoring.prepare import prepare_image_prediction_batch
from scoring.predict import get_batch_predictions_from_onnx


def _load_onnx_session(
    class_labels_json, onnx_model_path
) -> Tuple[onnxruntime.InferenceSession, List[str]]:
    """Load model ONNX inference session.

    Load ONNX model and associated class labels name list to associate model
    prediction indices into an ONNX inference session.

    Args:
        class_labels_json (str): path to local object class labels JSON file
        onnx_model_path (str): onnx model binary path

    Returns:
        Tuple[onnxruntime.InferenceSession, List[str]]: a tuple of the loaded
        ONNX inference session and the class labels list
    """
    with open(class_labels_json) as f:
        class_names = json.load(f)
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider']
    )
    return session, class_names


def get_onnx_model_img_dims(
    onnx_session: onnxruntime.InferenceSession,
) -> Tuple[int, int]:
    """For a loaded ONNX model, get the expected image width and height to correctly perform inference.

    Args:
        onnx_session (onnxruntime.InferenceSession): loaded ONNX model

    Returns:
        Tuple[int, int]: (ONNX model image width, ONNX model image height)
    """
    batch, channel, height_onnx, width_onnx = onnx_session.get_inputs()[0].shape
    return width_onnx, height_onnx


def init():
    """TODO model package docstring."""
    logging.info("Init started")
    classes_json_file_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "labels.json")
    onnx_file_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.onnx")
    onnx_session, class_names = _load_onnx_session(classes_json_file_path, onnx_file_path)
    width_onnx, height_onnx = get_onnx_model_img_dims(onnx_session)

    logging.info("Loaded models in the memory")

    global inference_variables
    inference_variables = {}
    inference_variables["onnx_session"] = onnx_session
    inference_variables["class_names"] = class_names
    inference_variables["width_onnx"] = width_onnx
    inference_variables["height_onnx"] = height_onnx
    logging.info("Init complete")


def run(raw_data):
    """TODO model package docstring."""
    # convert base64 string to images
    logging.info("Received a request to images")

    request = json.loads(raw_data)
    prediction_image_bytes = []
    for encoded_image_data in request["images"]:
        imgdata = base64.b64decode(encoded_image_data)
        prediction_image_bytes.append(imgdata)

    batch_size = len(prediction_image_bytes)
    logging.info("Request contains {} image(s) for inference".format(batch_size))

    # Prepare the batch of images to send to ONNX model for prediction
    predictions_img_batch = prepare_image_prediction_batch(
        batch_image_files=prediction_image_bytes,
        model_img_width=inference_variables["width_onnx"],
        model_img_height=inference_variables["height_onnx"],
        batch_size=batch_size,
    )
    logging.info("Prepared the batch of images")

    # Get the model object predictions for each image in the batch
    bbox_predictions = get_batch_predictions_from_onnx(
        onnx_session=inference_variables["onnx_session"],
        img_data_batch=predictions_img_batch,
        model_img_width=inference_variables["width_onnx"],
        model_img_height=inference_variables["height_onnx"],
        object_class_names=inference_variables["class_names"],
        score_threshold=0.8,
    )
    logging.info("Done with the prediction: Results are {}", bbox_predictions)

    return bbox_predictions
