"""Convert all onnx models in input_dir and downcast the weights to fp16."""
from pathlib import Path
from shutil import copyfile
import onnx
from onnxconverter_common import float16
import typer


def convert_fp32_to_fp16(input_dir: Path, output_dir: Path):
    """Convert model.onnx in input_dir and downcast the weights to fp16.

    Also copies over the associated labels.json file.

    Args:
        input_dir (Path): directory that contains fp32 onnx model
        output_dir (Path): directory where downcasted fp16 onnx model is stored
         When this directory doesn't exist, target directory is created
    """
    onnx_file = Path(input_dir, 'train_artifacts/model.onnx')
    label_file = Path(input_dir, 'train_artifacts/labels.json')
    if not onnx_file.is_file():
        raise FileNotFoundError(
            f"Directory: {input_dir} had no .onnx files."
            " Conversion process is terminated."
        )

    output_dir = Path(output_dir, "train_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(onnx_file)
    print(f"{onnx_file} will be converted to fp16")
    model_fp16 = float16.convert_float_to_float16(model)
    saved_model_path = Path(output_dir, "model.onnx")
    onnx.save(model_fp16, saved_model_path)
    print(
        "Conversion was successful and"
        f" fp16 onnx model was saved at {saved_model_path}"
    )

    fp16_label_file = Path(output_dir, 'labels.json')
    copyfile(label_file, fp16_label_file)


if __name__ == "__main__":
    typer.run(convert_fp32_to_fp16)
