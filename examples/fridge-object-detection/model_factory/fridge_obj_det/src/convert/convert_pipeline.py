"""This script runs mlops/components/convert.yml.

this step will eventually be integrated into one AML pipeline that is under development
This is currently for testing purpose for individual AML component
"""
from azure.ai.ml import Input, MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

onnx_test_ds = Input(
    path="azureml://datastores/workspaceblobstore/paths/onnx_test_fp32"
)

convert_component = load_component(source="mlops/components/convert.yml")


@pipeline(
    default_compute="dev-pipeline",
)
def convert_pipeline(input: Input):
    """Run convert component."""
    convert_component(fp32_input_dir=input)
    # this output will be used when this is integrated with other components
    # convert_node.outputs.fp16_output_dir


# create a pipeline
pipeline_job = convert_pipeline(input=onnx_test_ds)

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
    pipeline_job, experiment_name="model_conversion"
)
# wait until the job completes
ml_client.jobs.stream(pipeline_job.name)
