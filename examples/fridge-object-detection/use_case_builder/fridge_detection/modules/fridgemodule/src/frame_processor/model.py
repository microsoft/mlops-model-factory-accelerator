from typing import List


class InferenceRequest:
    """
    Class to hold the inference request
    """

    def __init__(self, frame: str) -> None:
        """
        Initialize InferenceRequest.

        @param
            frame str: Base64 encoded string of the frame to be sent in the request to inference module.
        """
        self.frame = frame


class InferenceResultRaw:
    """
    Class to hold the inference response
    """

    def __init__(self, box: dict, label: str, score: float) -> None:
        """
        Initialize InferenceResponse.

        @param
            box (dict): The bounding box co-ordinates of the object detected by inference module.
            label (str): The class of the object detected by inference module.
            score (float): The confidence of the object detected by inference module.
        """
        self.box = box
        self.label = label
        self.score = score


class InferenceResultTransformed:
    """
    Class for holding transformed inference result
    """

    def __init__(self, bb_boxes: List[InferenceResultRaw], object_cnt: dict) -> None:
        """
        Initialize InferenceResultTransformed.

        @param
            bb_boxes (List): The list of bounding boxes per frame returned by the inference module.
            object_cnt (dict): The dictionary of object counts per frame returned by the inference module.
        """
        self.bb_boxes = bb_boxes
        self.object_cnt = object_cnt
