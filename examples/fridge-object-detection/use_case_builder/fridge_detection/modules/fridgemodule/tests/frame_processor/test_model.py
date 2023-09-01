import unittest

from src.frame_processor.model import InferenceRequest, InferenceResultRaw, InferenceResultTransformed


class InferenceModelsTestCase(unittest.TestCase):

    def test_inference_request(self):
        frame = "base64_encoded_frame"
        inference_request = InferenceRequest(frame)
        self.assertEqual(inference_request.frame, frame)

    def test_inference_result_raw(self):
        box = {
            "topX": 0.1,
            "topY": 0.2,
            "bottomX": 0.3,
            "bottomY": 0.4
        }
        label = "object"
        score = 0.9
        inference_result_raw = InferenceResultRaw(box, label, score)
        self.assertEqual(inference_result_raw.box, box)
        self.assertEqual(inference_result_raw.label, label)
        self.assertEqual(inference_result_raw.score, score)

    def test_inference_result_transformed(self):
        bb_boxes = [
            InferenceResultRaw({"topX": 0.1, "topY": 0.2, "bottomX": 0.3, "bottomY": 0.4}, "object1", 0.9),
            InferenceResultRaw({"topX": 0.2, "topY": 0.3, "bottomX": 0.4, "bottomY": 0.5}, "object2", 0.8)
        ]
        object_cnt = {"object1": 1, "object2": 1}
        inference_result_transformed = InferenceResultTransformed(bb_boxes, object_cnt)
        self.assertEqual(inference_result_transformed.bb_boxes, bb_boxes)
        self.assertEqual(inference_result_transformed.object_cnt, object_cnt)


if __name__ == '__main__':
    unittest.main()
