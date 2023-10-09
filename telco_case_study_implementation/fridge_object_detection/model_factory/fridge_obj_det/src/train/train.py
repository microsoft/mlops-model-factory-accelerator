"""Train an AutoML object detection model for the fridge objects dataset."""
import argparse
from azure.ai.ml import Input, automl
import mlflow
from mlflow.tracking.client import MlflowClient
from common.mlops.get_aml_client import get_aml_client
import logging


def main(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    training_mltable_path: str,
    validation_mltable_path: str,
    automl_experiment_name: str,
    compute_cluster_name: str,
    automl_obj_det_model_name: str,
    model_artifacts_dir: str,
):
    """Train AutoML model and save the ONNX model artifacts after it finishes.

    Args:
        client_id (str): AAD client ID.
        client_secret (str): AAD client secret.
        tenant_id (str): AAD tenant ID.
        subscription_id (str): AML subscription ID.
        resource_group_name (str): AML resource group name.
        workspace_name (str): AML workspace name.
        training_mltable_path (str): training split MLTable path, could be local or
         registered data asset in AML.
        validation_mltable_path (str): validation split MLTable path, could be local or
         registered data asset in AML.
        automl_experiment_name (str): the experiment name used for AutoML training.
        compute_cluster_name (str): the compute cluster name used for AutoML training.
        automl_obj_det_model_name (str): the model variant used for AutoML training.
        model_artifacts_dir (str): the directory to save the model artifacts - at the end
         of model training by AutoML the ONNX model and the associated labels.json file will
         be saved to this directory.
    """
    try:
        # Create ML Client
        ml_client = get_aml_client(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        training_data_input = Input(type="mltable", path=training_mltable_path)
        validation_data_input = Input(type="mltable", path=validation_mltable_path)

        # Submit AutoML object detection experiment that will train a single model type
        image_object_detection_job = automl.image_object_detection(
            compute=compute_cluster_name,
            experiment_name=automl_experiment_name,
            training_data=training_data_input,
            validation_data=validation_data_input,
            target_column_name="label",
            primary_metric="mean_average_precision",
        )

        image_object_detection_job.set_training_parameters(
            model_name=automl_obj_det_model_name
        )

        # Submit the AutoML job
        returned_job = ml_client.jobs.create_or_update(image_object_detection_job)
        print(f"Created job: {returned_job}")

        ml_client.jobs.stream(returned_job.name)

        mlflow_tracking_uri = mlflow.get_tracking_uri()
        print(f"\nCurrent MLFlow tracking uri: {mlflow_tracking_uri}")
        mlflow_client = MlflowClient()

        job_name = returned_job.name

        mlflow_parent_run = mlflow_client.get_run(job_name)

        print("Parent Run: ")
        print(mlflow_parent_run)

        best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
        print(f"Found best child run id: {best_child_run_id}")

        best_run = mlflow_client.get_run(best_child_run_id)

        print("Best child run: ")
        print(best_run)

        _ = mlflow.artifacts.download_artifacts(
            run_id=best_run.info.run_id,
            artifact_path="train_artifacts/labels.json",
            dst_path=model_artifacts_dir,
        )
        _ = mlflow.artifacts.download_artifacts(
            run_id=best_run.info.run_id,
            artifact_path="train_artifacts/model.onnx",
            dst_path=model_artifacts_dir,
        )
        print(f"Artifacts downloaded in: {model_artifacts_dir}")
    except Exception as ex:
        print(f"Exception occurred in train main function: {ex}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_id", type=str, help="Azure client id")
    parser.add_argument("--client_secret", type=str, help="Azure client secret")
    parser.add_argument("--tenant_id", type=str, help="Azure tenant id")

    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument(
        "--resource_group_name", type=str, help="Azure Machine learning resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, help="Azure Machine learning Workspace name"
    )
    parser.add_argument(
        "--training_mltable_path",
        type=str,
        help="Training data MLTable path.",
        default="azureml:fridge-objects-train-mltable:latest",
    )
    parser.add_argument(
        "--validation_mltable_path",
        type=str,
        help="Validation data MLTable path.",
        default="azureml:fridge-objects-val-mltable:latest",
    )
    parser.add_argument(
        "--automl_compute_cluster_name",
        type=str,
        help="Compute cluster name to run the AutoML job on.",
        default="gpu-cluster-v100",
    )
    parser.add_argument(
        "--automl_experiment_name",
        type=str,
        help="AutoML experiment name.",
        default="automl-fridge-objects-detection-experiment",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="AutoML object detection model variant to train on.",
        default="fasterrcnn_resnet18_fpn",
    )
    parser.add_argument(
        "--automl_obj_det_model_name",
        type=str,
        help="fasterrcnn_resnet18_fpn",
    )

    parser.add_argument(
        "--model_artifacts_dir",
        type=str,
        help="This directory will contain AutoML ONNX file and labels.json file under traun_artifacts directory",
    )

    args = parser.parse_args()
    logging.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    main(
        args.client_id,
        args.client_secret,
        args.tenant_id,
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.training_mltable_path,
        args.validation_mltable_path,
        args.automl_experiment_name,
        args.automl_compute_cluster_name,
        args.automl_obj_det_model_name,
        args.model_artifacts_dir,
    )
