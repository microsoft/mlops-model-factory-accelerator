"""Pascal VOC to JSONL conversion for object detection annotations."""
import os
from xml.etree import ElementTree
import json


class JSONLConverter:
    """
    Base class for JSONL converters.

    ...
    Attributes
    ---------
    base_url : str
        the base for the image_url to be written into the jsonl file
    """

    def __init__(self, base_url: str):
        """Construct JSONLConverter.

        Args:
            base_url (str): the base for the image_url to be written into the jsonl file.
        """
        self.jsonl_data = []
        self.base_url = base_url

    def convert(self):
        """Inheriters should implement this method.

        Raises:
            NotImplementedError: when called on base class directly.
        """
        raise NotImplementedError


def write_json_lines(converter: JSONLConverter, filename: str):
    """Convert and write a JSONL file.

    Parameters:
        converter (JSONLConverter): the converter use to generate the jsonl
        filename (str): output file for writing jsonl
    """
    json_lines_data = converter.convert()
    with open(filename, "w") as outfile:
        for json_line in json_lines_data:
            json.dump(json_line, outfile, separators=(",", ":"))
            outfile.write("\n")
        print(f"Conversion completed. Converted {len(json_lines_data)} lines.")


class VOCJSONLConverter(JSONLConverter):
    """Class for converting VOC data for object detection into jsonl files."""

    def __init__(self, base_url: str, xml_dir: str):
        """Create VOCJSONLConverter.

        ...
        Attributes
        ---------
        base_url : str
            the base for the image_url to be written into the jsonl file
        xml_dir : str
            directory of xml annotation files
        """
        super().__init__(base_url=base_url)
        self.xml_dir = xml_dir

    def convert(self):
        """Generate jsonl data for object detection or instance segmentation.

        return: list of lines for jsonl
        rtype: List <class 'dict'>

        """
        json_line_sample = {
            "image_url": self.base_url,
            "image_details": {"format": None, "width": None, "height": None},
            "label": [],
        }

        for i, filename in enumerate(os.listdir(self.xml_dir)):
            if not filename.endswith(".xml"):
                print(f"Skipping unknown file: {filename}")
                continue

            annotation_filename = os.path.join(self.xml_dir, filename)
            print(f"Parsing {annotation_filename}")

            root = ElementTree.parse(annotation_filename).getroot()
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            labels = []
            for index, object in enumerate(root.findall("object")):
                name = object.find("name").text
                is_crowd = int(object.find("difficult").text)

                xmin = object.find("bndbox/xmin").text
                ymin = object.find("bndbox/ymin").text
                xmax = object.find("bndbox/xmax").text
                ymax = object.find("bndbox/ymax").text

                labels.append(
                    {
                        "label": name,
                        "topX": float(xmin) / width,
                        "topY": float(ymin) / height,
                        "bottomX": float(xmax) / width,
                        "bottomY": float(ymax) / height,
                        "isCrowd": is_crowd,
                    }
                )

            # build the jsonl file
            image_filename = root.find("filename").text
            _, file_extension = os.path.splitext(image_filename)
            json_line = dict(json_line_sample)
            json_line["image_url"] = os.path.join(
                json_line["image_url"], image_filename
            )
            json_line["image_details"]["format"] = file_extension[1:]
            json_line["image_details"]["width"] = width
            json_line["image_details"]["height"] = height
            json_line["label"] = labels

            self.jsonl_data.append(json_line)
        return self.jsonl_data
