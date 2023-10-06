"""TODO model docstring."""
import onnxruntime
from typing import Dict, List
import logging


def get_batch_predictions_from_onnx(
    onnx_session: onnxruntime.InferenceSession,
    img_data_batch,
    model_img_width: int,
    model_img_height: int,
    object_class_names: List[str],
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
                'label': bounding box class name,
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
        object_class_names (List[str]): Ordered list of object class names, will map model
         prediction indices to this list to get predicted object class names.
        score_threshold (float): confidence score threshold to filter predictions.
         Defaults to 0.8.

    Returns:
        (List[List[Dict]]): List of bounding box predictions per image in
         img_data_batch.
    """
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()

    output_names = [output.name for output in sess_output]

    batch_predictions = []
    for img_data in img_data_batch:
        try:
            predictions = onnx_session.run(
                output_names=output_names, input_feed={sess_input[0].name: [img_data]}
            )
            batch_predictions.append(predictions)
        except BaseException as error:
            logging.error('Error at %s', 'while running predictions using onnxruntime', exc_info=error)
            raise

    logging.info("batch predictions completed, no. of predictions: %s", len(batch_predictions))
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
            (model_img_height, model_img_width),
            object_class_names,
        )
        filtered_bounding_boxes = [
            box for box in bounding_boxes if box["score"] >= score_threshold
        ]
        filtered_boxes_batch.append(filtered_bounding_boxes)
    logging.info("No. of filtered predictions: %s", len(filtered_boxes_batch))
    return filtered_boxes_batch


def _get_box_dims(image_shape, box):
    box_keys = ["topX", "topY", "bottomX", "bottomY"]
    height, width = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims["topX"] = box_dims["topX"] * 1.0 / width
    box_dims["bottomX"] = box_dims["bottomX"] * 1.0 / width
    box_dims["topY"] = box_dims["topY"] * 1.0 / height
    box_dims["bottomY"] = box_dims["bottomY"] * 1.0 / height
    return box_dims


def _get_prediction(boxes, labels, scores, image_shape, classes):
    bounding_boxes = []
    for box, label_index, score in zip(boxes, labels, scores):
        box_dims = _get_box_dims(image_shape, box)

        box_record = {
            "box": box_dims,
            "label": classes[label_index],
            "score": score.item(),
        }

        bounding_boxes.append(box_record)

    return bounding_boxes
