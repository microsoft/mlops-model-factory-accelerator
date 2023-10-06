"""Returns an MLClient object using the provided credentials."""

import os
from azure.identity import EnvironmentCredential
from azure.ai.ml import MLClient


def get_aml_client(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
):
    """Create MLClient object using the provided credentials."""
    client = None
    try:
        os.environ["AZURE_CLIENT_ID"] = client_id
        os.environ["AZURE_CLIENT_SECRET"] = client_secret
        os.environ["AZURE_TENANT_ID"] = tenant_id

        credential = EnvironmentCredential()
        client = MLClient(
            credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        if client is None:
            raise Exception("Unable to create MLClient object.")
        return client
    except Exception as ex:
        print(f"Exception while creating MLClient: {ex}")
        raise
