"""Retrieve Azure Machine Learning workspace."""
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from common.logging.logger import get_logger

logger = get_logger("common_mlops")


def get_workspace(subscription_id: str, resource_group_name: str, workspace_name: str):
    """Get AML workspace.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name

    Returns:
        _type_: Workspace
    """
    try:
        logger.info(f"Getting access to {workspace_name} workspace.")
        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        workspace = client.workspaces.get(workspace_name)
        logger.info(f"Reference to {workspace_name} has been obtained.")
        return workspace
    except Exception:
        logger.exception(f"Not able to access workspace")
        raise


def main():
    """Return AML workspace."""
    parser = argparse.ArgumentParser("get_workspace")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument(
        "--resource_group_name", type=str, help="Azure Machine learning resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, help="Azure Machine learning Workspace name"
    )

    args = parser.parse_args()
    get_workspace(args.subscription_id, args.resource_group_name, args.workspace_name)


if __name__ == "__main__":
    main()
