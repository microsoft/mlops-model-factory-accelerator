"""Return AML Compute instance."""
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from common.logging.logger import get_logger

logger = get_logger("common_mlops")


def get_compute(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    cluster_name: str,
    cluster_size: str,
    cluster_region: str,
    min_instances: int,
    max_instances: int,
    idle_time_before_scale_down: int,
):
    """Return AML Compute instance.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        cluster_name (str): cluster name
        cluster_size (str): cluster size
        cluster_region (str): cluster region
        min_instances (int): min instances
        max_instances (int): max instances
        idle_time_before_scale_down (int): idle time

    Returns:
        _type_: AML Compute
    """
    compute_object = None
    try:
        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )
        compute_object = client.compute.get(cluster_name)
        logger.info(f"Found existing compute target {cluster_name}, so using it.")
    except ResourceNotFoundError:
        logger.info(f"{cluster_name} is not found! Trying to create a new one.")
        compute_object = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size=cluster_size,
            location=cluster_region,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time_before_scale_down,
        )
        logger.info("Able to create AML Compute object")
        compute_object = client.compute.begin_create_or_update(
            compute_object
        ).result()
        logger.info(f"A new cluster {cluster_name} has been created.")
    except Exception:
        logger.exception(f"Not able to access compute, exception")
        raise
    return compute_object


def main():
    """Return AML Compute instance."""
    parser = argparse.ArgumentParser("get_compute")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument(
        "--resource_group_name", type=str, help="Azure Machine learning resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, help="Azure Machine learning Workspace name"
    )
    parser.add_argument(
        "--cluster_name", type=str, help="Azure Machine learning cluster name"
    )
    parser.add_argument(
        "--cluster_size", type=str, help="Azure Machine learning cluster size"
    )
    parser.add_argument(
        "--cluster_region", type=str, help="Azure Machine learning cluster region"
    )
    parser.add_argument("--min_instances", type=int, default=0)
    parser.add_argument("--max_instances", type=int, default=4)
    parser.add_argument("--idle_time_before_scale_down", type=int, default=120)

    args = parser.parse_args()
    get_compute(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.cluster_name,
        args.cluster_size,
        args.cluster_region,
        args.min_instances,
        args.max_instances,
        args.idle_time_before_scale_down,
    )


if __name__ == "__main__":
    main()
