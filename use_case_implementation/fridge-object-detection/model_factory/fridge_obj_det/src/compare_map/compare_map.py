"""Convert all onnx models in input_dir and downcast the weights to fp16."""
from warnings import warn
from decimal import Decimal

import json
import typer
from pathlib import Path


def extract_map(
        results_file: Path
) -> float:
    """Extract mAP@0.5 from results.json file.

    Args:
        results_file (Path): Path to results file containing
            results.json file with mAP metrics.

    Returns:
        float: mAP@0.5 score
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    map_score = results['map_50']
    return map_score


def create_metrics_json(
    map_before: float,
    map_after: float,
    metrics_json_file: Path
):
    """Create metrics json file with mAP scores.

    Args:
        map_before (float): fp32 mAP score.
        map_after (float): fp16 mAP score.
        metrics_json_file (Path): Path to metrics json file for writing.
    """
    map_dict = {
        "map_onnx_fp16": map_after,
        "map_onnx_fp32": map_before}
    json_content = json.dumps(map_dict)
    with open(metrics_json_file, "w") as f:
        f.write(json_content)


def compare_scores(
    map_before: float,
    map_after: float,
    tolerance: float = 0.01,
    throws_error: bool = False,
):
    """Compare mAP before and after onnx fp16 conversion and raise ValueError or print warning if mAP drop is beyond tolerance.

    Args:
        map_before (float): mAP metric before onnx fp16 conversion.
        map_after (float): mAP metric after onnx fp16 conversion.
        tolerance (float, optional): threshold to tolerate mAP value drop.
          map_before - map_after <= tolerance will be considered as acceptable.
          Defaults to 0.01.
        throws_error (bool, optional): whether to throw error when mAP drop is
          beyond tolerance. When this is off, it will print warning instead.
          Defaults to False.

    Raises:
        ValueError: raised when throws_error is True and mAP dropped beyond tolerance
    """
    # without Decimal,
    # >>> a = 0.98
    # >>> b = 0.97
    # >>> a-b
    # 0.010000000000000009
    # this causes a - b > 0.01 to be True:
    # >>> float(Decimal(str(a)) - Decimal(str(b))) == 0.01
    # True
    map_before = Decimal(str(map_before))
    map_after = Decimal(str(map_after))
    if float(map_before - map_after) > tolerance:
        if throws_error:
            raise ValueError(
                f"mAP dropped {map_before - map_after} beyond tolerance {tolerance}."
                f" mAP before conversion: {map_before},"
                f" mAP after conversion: {map_after}"
            )
        else:
            warn(
                f"mAP dropped {map_before - map_after} beyond tolerance {tolerance}."
                f" mAP before conversion: {map_before},"
                f" mAP after conversion: {map_after}"
            )
    if 0 <= float(map_before - map_after) <= tolerance:
        print(
            f"mAP dropped {map_before - map_after} within tolerance {tolerance}."
            f" mAP before conversion: {map_before}, mAP after conversion: {map_after}"
        )
    if float(map_before - map_after) < 0:
        print(
            f"mAP increased {abs(map_before - map_after)}."
            f" mAP before conversion: {map_before}, mAP after conversion: {map_after}"
        )


def compare_map_before_and_after_conversion(
    fp32_results_file: Path,
    fp16_results_file: Path,
    metrics_json_file: Path,
    tolerance: float = 0.01,
    throws_error: bool = False,
):
    """Compare mAP before and after onnx fp16 conversion and raise ValueError or print warning if mAP drop is beyond tolerance.

    Reads metrics files from scoring steps and extracts mAP@0.5 before comparing
    them. Raises an error/prints a warning if the mAP drop is beyond tolerance,
    and writes a json file with these metrics.

    Args:
        fp32_results_file (Path): mAP metrics before onnx fp16 conversion
        fp16_results_file (Path): mAP metrics after onnx fp16 conversion
        metrics_json_file (str): Path to metrics json file for writing.
        tolerance (float, optional): threshold to tolerate mAP value drop.
          map_before - map_after <= tolerance will be considered as acceptable.
          Defaults to 0.01.
        throws_error (bool, optional): whether to throw error when mAP drop is
          beyond tolerance. When this is off, it will print warning instead.
          Defaults to False.

    Raises:
        ValueError: raised when throws_error is True and mAP dropped beyond tolerance
    """
    map_before = extract_map(fp32_results_file)
    map_after = extract_map(fp16_results_file)

    compare_scores(map_before, map_after, tolerance, throws_error)

    create_metrics_json(map_before, map_after, metrics_json_file)


if __name__ == "__main__":
    typer.run(compare_map_before_and_after_conversion)
