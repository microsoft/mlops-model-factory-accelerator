"""Register the ONNX model to AML workspace."""
import argparse
import json
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from common.mlops.get_aml_client import get_aml_client
import logging
import shutil


def main(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    input_model_artifacts_path: str,
    registered_model_name: str,
    registered_model_description: str,
    build_reference: str,
    metrics_json_path: str
):
    """Register the ONNX model to the AML workspace.

    Args:
        client_id (str): AAD client ID.
        client_secret (str): AAD client secret.
        tenant_id (str): AAD tenant ID.
        subscription_id (str): AML subscription ID.
        resource_group_name (str): AML resource group name.
        workspace_name (str): AML workspace name.
        input_model_artifacts_path (str): the path to the input model artifacts. Should
         contain the model ONNX file and the labels.json file.
        registered_model_name (str): the name of the registered model in AML.
        registered_model_description (str): the description of the registered model in AML.
        build_reference (str): the AzDO build reference that generated the model.
        metrics_json_path (str): the path to the metrics.json file containing the mAP score
         on the test set for the ONNX FP32 model and the ONNX FP16 model.
    """
    # Create ML Client
    ml_client = get_aml_client(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    if ml_client is None:
        raise Exception("Could not create MLClient")

    print(f"ML Client created successfully {str(ml_client)}")

    # load the metrics file and get the mAP scores
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)
        map_onnx_fp16 = metrics["map_onnx_fp16"]
        map_onnx_fp32 = metrics["map_onnx_fp32"]

    compressed_model_file = shutil.make_archive(
        base_name="model_artifacts", format="gztar", root_dir=input_model_artifacts_path
    )

    onnx_model = Model(
        path=compressed_model_file,
        name=registered_model_name,
        description=registered_model_description,
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "build_reference": build_reference,
            "map_onnx_fp16": map_onnx_fp16,
            "map_onnx_fp32": map_onnx_fp32,
        },
    )
    registered_model = ml_client.models.create_or_update(onnx_model)
    logging.info(f"The registered model ID: {registered_model.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, help="Azure client id")
    parser.add_argument("--client_secret", type=str, help="Azure client secret")
    parser.add_argument("--tenant_id", type=str, help="Azure tenant id")

    parser.add_argument("--subscription_id", type=str,
                        help="Azure subscription id")
    parser.add_argument(
        "--resource_group_name", type=str, help="Azure Machine learning resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, help="Azure Machine learning Workspace name"
    )
    parser.add_argument(
        "--input_model_artifacts_path",
        type=str,
        help="The path to the input model artifacts. Should include the ONNX model and the labels.json file.",
    )
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default="fridge-objects-automl-onnx",
        help="The name of the registered model.",
    )
    parser.add_argument(
        "--registered_model_description",
        type=str,
        default="Best AutoML Object Detection ONNX model for fridge objects dataset.",
        help="The description of the registered model.",
    )
    parser.add_argument(
        "--build_reference",
        type=str,
        help="Original AzDo build id that initiated experiment",
    )
    parser.add_argument(
        "--metrics_json_path",
        type=str,
        help="Path to the metrics.json file containing the mAP scores for the ONNX FP32 and FP16 models.",
    )
    args = parser.parse_args()
    main(
        client_id=args.client_id,
        client_secret=args.client_secret,
        tenant_id=args.tenant_id,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
        input_model_artifacts_path=args.input_model_artifacts_path,
        registered_model_name=args.registered_model_name,
        registered_model_description=args.registered_model_description,
        build_reference=args.build_reference,
        metrics_json_path=args.metrics_json_path,
    )
