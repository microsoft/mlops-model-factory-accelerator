"""This script runs mlops/components/compare_map.yml.

this step will eventually be integrated into one AML pipeline that is under development
This is currently for testing purpose for individual AML component
"""
from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

compare_map_component = load_component(source="mlops/components/compare_map.yml")


@pipeline(
    default_compute="dev-pipeline",
)
def compare_map_pipeline(map_before, map_after):
    """Run compare_map component."""
    compare_map_component(map_before=map_before, map_after=map_after)


# create a pipeline
# TODO: these input values will be replaced
# with the actual computed values from previous steps
pipeline_job = compare_map_pipeline(map_before=0.98, map_after=0.97)

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="map_comparison"
)
# wait until the job completes
ml_client.jobs.stream(pipeline_job.name)
