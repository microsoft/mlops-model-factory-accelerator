"""Run onnx inferencing on a test set and calculate mAP score.

Code for this script has been adopted from:
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?view=azureml-api-2&tabs=object-detect-cnn
"""
from typing import Dict, List, Tuple
import mltable
import json
import onnxruntime
import numpy as np
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import typer
from pathlib import Path
import onnx


def load_mltable(
        mltable_folder: str
) -> Tuple[List[str], List[np.array]]:
    """Load MLTable and extract lists of images and labels.

    Args:
        mltable_folder (str): path to folder containing MLTable yaml and .jsonl file.
            Can be local or in Azure blob storage.

    Returns:
        List[str]: list of image paths in StreamInfo format
        List[np.array(Dict)]: list of arrays of dictionaries containing label bboxes
            and classes
    """
    tbl = mltable.load(mltable_folder)
    df = tbl.to_pandas_dataframe()
    images = df['image_url'].tolist()
    labels = df['label'].tolist()
    return images, labels


def load_onnx_session(
        class_labels_json: str,
        onnx_model_path: str
) -> Tuple[onnxruntime.InferenceSession, List[str]]:
    """Load model ONNX inference session.

    Also loads associated class labels name
    list to associate model prediction indices with.

    Args:
        class_labels_json (str): path to local object class labels JSON file
        onnx_model_path (str): onnx model binary path

    Returns:
        Tuple[onnxruntime.InferenceSession, List[str]]: a tuple of the loaded
        ONNX inference session and the class labels list
    """
    onnx_model = onnx.load(onnx_model_path)
    print(onnx.checker.check_model(onnx_model))
    print('onnxruntime device: ', onnxruntime.get_device())
    with open(class_labels_json) as f:
        class_names = json.load(f)
    print("Model ordered object class name:")
    print(class_names)
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=['CUDAExecutionProvider']
    )
    print("ONNX model loaded...")

    return session, class_names


def get_onnx_model_img_dims(
    onnx_session: onnxruntime.InferenceSession,
) -> Tuple[int, int]:
    """Get the expected image width and height to correctly perform inference.

    Args:
        onnx_session (onnxruntime.InferenceSession): loaded ONNX model

    Returns:
        Tuple[int, int]: (ONNX model image width, ONNX model image height)
    """
    batch, channel, height_onnx, width_onnx = onnx_session.get_inputs()[0].shape
    return width_onnx, height_onnx


def preprocess_image_for_prediction(image_path: str, height_onnx: int, width_onnx: int):
    """Perform pre-processing on raw input image.

    Transform, resize and normalize the image for expected Faster-RCNN ONNX prediction.

    Args:
        image_path (str): path to image in StreamInfo format.
        height_onnx (int): ONNX model expected image height.
        width_onnx (int): ONNX model expected image width.

    Returns:
        ndarray: Pre-processed image in numpy format, shape: 1xCxHxW
    """
    with image_path.open() as f:
        image = Image.open(f)
        image = image.convert("RGB")
        image = image.resize((width_onnx, height_onnx))
        np_image = np.array(image)
    # HWC -> CHW
    np_image = np_image.transpose(2, 0, 1)  # CxHxW
    # normalize the image
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype("float32")
    # TODO: should check if float32 is compatible with f16 onnx models
    for i in range(np_image.shape[0]):
        norm_img_data[i, :, :] = (np_image[i, :, :] / 255 - mean_vec[i]) / std_vec[i]
    np_image = np.expand_dims(norm_img_data, axis=0)  # 1xCxHxW
    return np_image


def prepare_image_prediction_batch(
    batch_image_files: List[str],
    model_img_width: int,
    model_img_height: int,
    batch_size: int,
) -> List:
    """Pre-process list of image filenames into a batch.

    The result of this function can be passed to
    get_batch_predictions_from_ONNX() for inference results on each image.

    Args:
        batch_image_files (List[str]): local file paths to images, length must be
         same as batch_size
        model_img_width (int): expected ONNX model input image width
        model_img_height (int): expected ONNX model input image height
        batch_size (int): batch size of images to prepare, should be equal
         to len(batch_image_files)

    Returns:
        List[ndarray]: pre-processed image batch with length == batch_image_files
    """
    img_processed_list = []
    for i in range(batch_size):
        img_processed_list.append(
            preprocess_image_for_prediction(batch_image_files[i], model_img_height, model_img_width)
        )

    if len(img_processed_list) > 1:
        img_data = np.concatenate(img_processed_list)
    elif len(img_processed_list) == 1:
        img_data = img_processed_list[0]
    else:
        img_data = None

    assert batch_size == img_data.shape[0]

    return img_data


def _get_box_dims(
        image_shape: Tuple[float],
        box: torch.Tensor
        ) -> Dict:
    """Scale box dimensions from absolute to relative.

    Args:
        image_shape (Tuple): Image width and height
        box (torch.Tensor): Array of absolute bounding box coordinates

    Returns:
        box_dims (Dict): Relative box coordinates
    """
    box_keys = ["topX", "topY", "bottomX", "bottomY"]
    height, width = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims["topX"] = box_dims["topX"] * 1.0 / width
    box_dims["bottomX"] = box_dims["bottomX"] * 1.0 / width
    box_dims["topY"] = box_dims["topY"] * 1.0 / height
    box_dims["bottomY"] = box_dims["bottomY"] * 1.0 / height

    return box_dims


def _get_prediction(
        boxes: torch.Tensor,
        labels: List[int],
        scores: torch.Tensor,
        image_shape: Tuple[float]
        ) -> List[Dict]:
    """Collate bbox predictions into list.

    Args:
        boxes (torch.Tensor): 2D array of bbox predictions
            for an image
        labels (List[int]): List of class predictions
        scores (torch.Tensor): List of prediction scores
        image_shape (Tuple[float]): Image dimensions

    Returns:
        List[Dict]: List of bounding boxes in an image, with
            each element containing a dictionary of box
            coordinates, the label and the score.
    """
    bounding_boxes = []
    for box, label_index, score in zip(boxes, labels, scores):
        box_dims = _get_box_dims(image_shape, box)

        box_record = {
            "box": box_dims,
            "label": label_index,
            "score": score.item(),
        }

        bounding_boxes.append(box_record)

    return bounding_boxes


def get_batch_predictions_from_onnx(
    onnx_session: onnxruntime.InferenceSession,
    img_data_batch,
    model_img_width: int,
    model_img_height: int,
    score_threshold: float = 0.8,
) -> List[List[Dict]]:
    """Perform predictions with ONNX runtime for a batch of images.

    Returns a list for each image in img_data_batch, where each list per image
    is a list of bounding box predictions (dict) with the following structure:

    [ # list length of img_data_batch
        [ # per image list of bounding box predictions
            {
                'box': {
                    'topX': normalised top left bounding box X co-ordinate
                    'topY': normalised top left bounding box Y co-ordinate
                    'bottomX': normalised bottom right bounding box X co-ordinate
                    'bottomY': normalised bottom right bounding box Y co-ordinate
                },
                'label': bounding box class index,
                'score': bounding box confidence score
            }
        ]
    ]

    Note that bounding box co-ordinates are normalised to the range [0, 1] to allow
    scaling to the original image size (as the original image may have beeen resized for
    model prediction).

    Args:
        onnx_session (onnxruntime.InferenceSession): the ONNX runtime inference session
         with model loaded.
        img_data_batch (List[ndarray]): pre-processed list of images ready for prediction, each image
         should have shape CxHxW.
        model_img_width (int): ONNX model image width
        model_img_height (int): ONNX model image height
        score_threshold (float): confidence score threshold to filter predictions.
         Defaults to 0.8.

    Returns:
        (List[List[Dict]]): List of bounding box predictions per image in
         img_data_batch.
    """
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()

    # predict with ONNX Runtime
    output_names = [output.name for output in sess_output]
    batch_predictions = []
    for img_data in img_data_batch:
        predictions = onnx_session.run(
            output_names=output_names, input_feed={sess_input[0].name: [img_data]}
        )
        batch_predictions.append(predictions)

    # Filter the results with threshold.
    filtered_boxes_batch = []
    for batch_sample in batch_predictions:
        # in case of retinanet change the order of boxes, labels, scores to boxes, scores, labels
        # confirm the same from order of boxes, labels, scores output_names
        boxes, labels, scores = batch_sample[0], batch_sample[1], batch_sample[2]
        bounding_boxes = _get_prediction(
            boxes,
            labels,
            scores,
            (model_img_height, model_img_width)
        )
        filtered_bounding_boxes = [
            box for box in bounding_boxes if box["score"] >= score_threshold
        ]
        filtered_boxes_batch.append(filtered_bounding_boxes)

    return filtered_boxes_batch


def convert_bbox_predictions_to_torch(
        bboxes: List[List[Dict]]
) -> List[Dict]:
    """Pre-process bounding box predictions from ONNX into a format that is ready to be passed into torchmetrics.

    Args:
        bboxes (List[List[Dict]]): list of bounding box predictions
            from ONNX

    Returns:
        List[Dict]: list of bounding box predictions where each
            element is a dictionary representing the predicted
            bboxes, labels, and scores for a single image. Values
            inside the dictionary are torch.Tensor objects.
    """
    coords = ['topX', 'topY', 'bottomX', 'bottomY']
    torch_bboxes = []

    for image in bboxes:
        boxes = []
        labels = []
        scores = []
        for detection in image:
            bbox_list = [detection['box'][coord] for coord in coords]
            boxes.append(bbox_list)
            labels.append(detection['label'])
            scores.append(detection['score'])
        new_image = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.Tensor(labels),
            'scores': torch.FloatTensor(scores)
        }
        torch_bboxes.append(new_image)
    return torch_bboxes


def convert_annotations_to_torch(
        annotation_list: List[np.array],
        classes: List[str]
) -> List[Dict]:
    """Pre-process bounding box annotations from .jsonl format to a format that is ready to be passed into torchmetrics.

    Args:
        annotation_list (List[np.array(Dict)]): list of arrays containing
            a dictionary for each bbox and class.
        classes (List[str]): list of class names for encoding into
            integer indices.

    Returns:
        List[Dict]: list of bounding box predictions where each
            element is a dictionary representing the predicted
            bboxes and labels for a single image. Values
            inside the dictionary are torch.Tensor objects.
    """
    coords = ['topX', 'topY', 'bottomX', 'bottomY']
    torch_bboxes = []

    for label_array in annotation_list:
        boxes = []
        labels = []
        for label in label_array:
            bbox_list = [label[coord] for coord in coords]
            boxes.append(bbox_list)
            labels.append(classes.index(label['label']))

        new_label = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.Tensor(labels)
        }
        torch_bboxes.append(new_label)
    return torch_bboxes


def calculate_map_score(
        predictions: List[Dict],
        annotations: List[Dict],
        results_file: str
):
    """
    Calculate the mAP@0.5 by comparing the predicted and annotated bounding boxes and classes.

    Args:
        predictions (List[Dict]): list of bounding box predictions
            from ONNX, pre-processed into torch format.
        annotations (List[Dict]): list of bounding box annotations
            pre-processed into torch format.
        results_file (str): destination path for the results file
            containing mAP scores (map, map_50, map_per_class, etc.)
    """
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(predictions, annotations)
    results = metric.compute()
    results = {key: value.tolist() for key, value in results.items()}
    print(results)

    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Saved results at {results_file}')


def inference_and_map(
        model_folder: Path,
        mltable_folder: str,
        results_file: Path
):
    """Inference and comparison to ground truth labels via mAP calculation."""
    image_paths, annotation_list = load_mltable(mltable_folder)
    batch_size = len(image_paths)
    labels_file_path = str(Path(model_folder, "train_artifacts/labels.json"))
    onnx_model_path = str(Path(model_folder, "train_artifacts/model.onnx"))

    # Load the downloaded ONNX model and associated class names for inference
    onnx_session, class_names = load_onnx_session(
        class_labels_json=labels_file_path,
        onnx_model_path=onnx_model_path
    )

    # Get the expected model image width and height so we can preprocess img
    # correctly before passing to model
    width_onnx, height_onnx = get_onnx_model_img_dims(onnx_session)

    # Prepare the batch of images to send to ONNX model for prediction
    predictions_img_batch = prepare_image_prediction_batch(
        batch_image_files=image_paths,
        model_img_width=width_onnx,
        model_img_height=height_onnx,
        batch_size=batch_size,
    )

    # Get the model object predictions for each image in the batch
    bbox_predictions = get_batch_predictions_from_onnx(
        onnx_session=onnx_session,
        img_data_batch=predictions_img_batch,
        model_img_width=width_onnx,
        model_img_height=height_onnx,
        score_threshold=0.8,
    )

    # Convert the bbox predictions into torch format
    torch_predictions = convert_bbox_predictions_to_torch(
        bboxes=bbox_predictions
    )

    # Read the annotations into torch format
    torch_annotations = convert_annotations_to_torch(
        annotation_list=annotation_list,
        classes=class_names,
    )

    # Calculate the mAP score
    calculate_map_score(
        predictions=torch_predictions,
        annotations=torch_annotations,
        results_file=results_file
    )


if __name__ == "__main__":
    typer.run(inference_and_map)
