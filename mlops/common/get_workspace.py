from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import argparse
from mlops.common.logger import get_logger

logger = get_logger()

def get_workspace(subscription_id: str, resource_group_name: str, workspace_name: str):
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
    except Exception as ex:
        logger.exception("Oops!  invalid credentials.. Try again...")
        raise


def main():
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
