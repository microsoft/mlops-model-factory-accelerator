"""TODO model package docstring."""
from typing import List
from PIL import Image
import numpy as np
import io


def preprocess_image_for_prediction(image: Image, height_onnx: int, width_onnx: int):
    """Perform pre-processing on raw input image.

    Transform, resize and normalize the image for expected Faster-RCNN ONNX prediction.

    Args:
        image (Image): PIL.Image loaded image.
        height_onnx (int): ONNX model expected image height.
        width_onnx (int): ONNX model expected image width.

    Returns:
        ndarray: Pre-processed image in numpy format, shape: 1xCxHxW
    """
    image = image.convert("RGB")
    image = image.resize((width_onnx, height_onnx))
    np_image = np.array(image)
    # HWC -> CHW
    np_image = np_image.transpose(2, 0, 1)  # CxHxW
    # normalize the image
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype("float32")
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
    """Pre-process list of image filenames for inference.

    Transform list of images (of batch_size) into a batch that is ready to be passed into
    model prediction. The result of this function can be passed to get_batch_predictions_from_onnx()
    for inference results on each image.

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
        img = Image.open(io.BytesIO(batch_image_files[i]))
        img_processed_list.append(
            preprocess_image_for_prediction(img, model_img_height, model_img_width)
        )

    if len(img_processed_list) > 1:
        img_data = np.concatenate(img_processed_list)
    elif len(img_processed_list) == 1:
        img_data = img_processed_list[0]
    else:
        img_data = None

    assert batch_size == img_data.shape[0]

    return img_data
