"""This script reads conda.yml file and creates an environment from it.

Run python create_devenv.py at this directory to create the environment.
This is meant to be used during development
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="conda.yml",
    name="conda-based-devenv-py38-cpu",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client = MLClient.from_config(credential=credential)
ml_client.environments.create_or_update(env_docker_conda)
