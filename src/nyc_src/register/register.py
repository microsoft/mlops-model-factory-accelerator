import mlflow
import argparse
import os
import json
from pathlib import Path

from mlops.common.logger import get_logger

logger = get_logger("nyc_taxi_register")

def main(model_metadata, model_name, score_report, build_reference):
    try:
        run_file = open(args.model_metadata)
        model_metadata = json.load(run_file)
        run_uri = model_metadata["run_uri"]

        score_file = open(Path(args.score_report) / "score.txt")
        score_data = json.load(score_file)
        cod = score_data["cod"]
        mse = score_data["mse"]
        coff = score_data["coff"]

        model_version = mlflow.register_model(run_uri, model_name)

        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name=model_name, version=model_version.version, key="mse", value=mse
        )
        client.set_model_version_tag(
            name=model_name, version=model_version.version, key="coff", value=coff
        )
        client.set_model_version_tag(
            name=model_name, version=model_version.version, key="cod", value=cod
        )
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="build_id",
            value=build_reference,
        )

        logger.info(model_version)
    except Exception as ex:
        logger.info(ex)
        raise
    finally:
        run_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("register_model")
    parser.add_argument(
        "--model_metadata",
        type=str,
        help="model metadata on Machine Learning Workspace",
    )
    parser.add_argument("--model_name", type=str, help="model name to be registered")
    parser.add_argument("--score_report", type=str, help="score report for the model")
    parser.add_argument(
        "--build_reference",
        type=str,
        help="Original AzDo build id that initiated experiment",
    )

    args = parser.parse_args()

    logger.info(args.model_metadata)
    logger.info(args.model_name)
    logger.info(args.score_report)
    logger.info(args.build_reference)

    main(args.model_metadata, args.model_name, args.score_report, args.build_reference)
