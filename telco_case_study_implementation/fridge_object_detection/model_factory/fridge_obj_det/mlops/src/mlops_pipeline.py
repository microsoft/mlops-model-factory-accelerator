"""Fridge objects AML components based training pipeline."""
from typing import Optional
import os
import time
import logging
import argparse
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from common.mlops.get_compute import get_compute
from common.mlops.get_environment import get_environment
from common.mlops.get_aml_client import get_aml_client


gl_pipeline_components = []


@pipeline()
def fridge_objects_automl_train(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    automl_model_name: str,
    automl_experiment_name: str,
    automl_compute_cluster_name: str,
    build_reference_id: str,
    model_name: str,
    model_description: str,
    deploy_environment: str
) -> None:
    """Compose the fridge objects AutoML training pipeline.

    Adds steps for data preparation (creating train, val, test MLTables) and then launches
    an AutoML object detection training job.

    Args:
        subscription_id (str): AML subscription ID.
        resource_group_name (str): AML resource group name.
        workspace_name (str): AML workspace name.
        automl_model_name (str): the AutoML object detection model variant.
        automl_experiment_name (str): the AutoML experiment name.
        automl_compute_cluster_name (str): the compute cluster name to use
         for the AutoML job.
        build_reference_id (str): the DevOps build reference ID executing the pipeline.
        model_name (str): name of model shown at registration.
        model_description (str): description of model shown at registration.
        deploy_environment (str): the environment to use for the AutoML job.

    Returns:
        None
    """
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")

    if tenant_id is None or client_id is None or client_secret is None:
        raise ValueError("Env variables not set, unable to create client")

    train_mltable_name = "fride_obj_det_mltable_train_" + deploy_environment
    val_mltable_name = "fride_obj_det_mltable_val_" + deploy_environment
    test_mltable_name = "fride_obj_det_mltable_test_" + deploy_environment

    prepare_fridge_obj_data = gl_pipeline_components[0](
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        train_mltable_name=train_mltable_name,
        val_mltable_name=val_mltable_name,
        test_mltable_name=test_mltable_name,
    )

    train_automl_model = gl_pipeline_components[1](
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        training_mltable_path=prepare_fridge_obj_data.outputs.train_mltable,
        validation_mltable_path=prepare_fridge_obj_data.outputs.val_mltable,
        automl_obj_det_model_name=automl_model_name,
        automl_experiment_name=automl_experiment_name,
        automl_compute_cluster_name=automl_compute_cluster_name,
    )

    gl_pipeline_components[2](
        fp32_input_dir=train_automl_model.outputs.model_artifacts_dir
    )

    score_fp32_model = gl_pipeline_components[3](
        model_folder_path=train_automl_model.outputs.model_artifacts_dir,
        mltable_folder=prepare_fridge_obj_data.outputs.test_mltable
    )

    score_fp16_model = gl_pipeline_components[4](
        model_folder_path=train_automl_model.outputs.model_artifacts_dir,
        mltable_folder=prepare_fridge_obj_data.outputs.test_mltable
    )
    # TODO: change model input to convert_onnx_model.outputs.fp16_output_dir

    compare_map_scores = gl_pipeline_components[5](
        map_before=score_fp32_model.outputs.results_file,
        map_after=score_fp16_model.outputs.results_file
    )

    gl_pipeline_components[6](
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        onnx_model_artifacts_folder=train_automl_model.outputs.model_artifacts_dir,
        registered_model_name=model_name,
        registered_model_description=model_description,
        build_reference_id=build_reference_id,
        metrics_json_file=compare_map_scores.outputs.metrics_json_file
    )


def construct_pipeline(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    cluster_name: str,
    environment_name: str,
    model_name: str,
    model_description: str,
    display_name: str,
    deploy_environment: str,
    build_reference: str,
    automl_model_name: str,
    automl_experiment_name: str,
    automl_compute_cluster_name: str
) -> None:
    """Construct the AML components based pipeline.

    Args:
        subscription_id (str): AML subscription ID.
        resource_group_name (str): AML resource group name.
        workspace_name (str): AML workspace name.
        cluster_name (str): the AML cluster name used to run the pipeline steps.
        environment_name (str): the AML environment name used to run the pipeline steps.
        model_name (str): name of model shown at registration.
        model_description (str): description of model shown at registration.
        display_name (str): the display name of the pipeline run.
        deploy_environment (str): the stage of deployment (eg. dev, prod).
        build_reference (str): the DevOps build reference ID executing the pipeline.
        automl_model_name (str): the AutoML object detection model variant.
        automl_experiment_name (str): the AutoML experiment name.
        automl_compute_cluster_name (str): the AML compute cluster name to use to run the
         AutoML job.

    Returns:
        None
    """
    parent_dir = os.path.join(
        os.getcwd(), "fridge_obj_det/mlops/components"
    )

    prepare_data = load_component(source=parent_dir + "/prep.yml")
    train_model = load_component(source=parent_dir + "/train.yml")
    convert_model = load_component(source=parent_dir + "/convert.yml")
    score_fp32 = load_component(source=parent_dir + "/score.yml")
    score_fp16 = load_component(source=parent_dir + "/score.yml")
    compare_map = load_component(source=parent_dir + "/compare_map.yml")
    register_model = load_component(source=parent_dir + "/register.yml")

    # Set the environment name to custom environment using name and version number
    prepare_data.environment = environment_name
    train_model.environment = environment_name
    convert_model.environment = environment_name
    score_fp32.environment = environment_name
    score_fp16.environment = environment_name
    compare_map.environment = environment_name
    register_model.environment = environment_name

    gl_pipeline_components.append(prepare_data)
    gl_pipeline_components.append(train_model)
    gl_pipeline_components.append(convert_model)
    gl_pipeline_components.append(score_fp32)
    gl_pipeline_components.append(score_fp16)
    gl_pipeline_components.append(compare_map)
    gl_pipeline_components.append(register_model)

    pipeline_job = fridge_objects_automl_train(
        subscription_id,
        resource_group_name,
        workspace_name,
        automl_model_name,
        automl_experiment_name,
        automl_compute_cluster_name,
        build_reference,
        model_name,
        model_description,
        deploy_environment
    )
    pipeline_job.display_name = display_name
    pipeline_job.tags = {
        "environment": deploy_environment,
        "build_reference": build_reference,
    }

    # set pipeline level compute
    pipeline_job.settings.default_compute = cluster_name
    pipeline_job.settings.force_rerun = False
    # set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    return pipeline_job


def execute_pipeline(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    experiment_name: str,
    pipeline_job: pipeline,
    wait_for_completion: bool,
    output_file: Optional[str],
):
    """Execute the AML components based pipeline.

    Args:
        subscription_id (str): AML subscription ID.
        resource_group_name (str): AML resource group name.
        workspace_name (str): AML workspace name.
        experiment_name (str): AML pipeline experiment name.
        pipeline_job (pipeline): the AML pipeline to execute.
        wait_for_completion (bool): True if the function should wait for the
         pipeline to complete.
        output_file (Optional[str]): _description_

    Raises:
        Exception: _description_
    """
    try:
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")

        if tenant_id is None or client_id is None or client_secret is None:
            raise ValueError("Env variables not set, unable to create client")

        ml_client = get_aml_client(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name=experiment_name
        )

        logging.info(f"The job {pipeline_job.name} has been submitted!")
        if output_file is not None:
            with open(output_file, "w") as out_file:
                out_file.write(pipeline_job.name)

        if wait_for_completion is True:
            total_wait_time = 3600
            current_wait_time = 0
            job_status = [
                "NotStarted",
                "Queued",
                "Starting",
                "Preparing",
                "Running",
                "Finalizing",
                "Provisioning",
                "CancelRequested",
                "Failed",
                "Canceled",
                "NotResponding",
            ]

            while pipeline_job.status in job_status:
                if current_wait_time <= total_wait_time:
                    time.sleep(20)
                    pipeline_job = ml_client.jobs.get(pipeline_job.name)

                    current_wait_time = current_wait_time + 15

                    if (
                        pipeline_job.status == "Failed"
                        or pipeline_job.status == "NotResponding"
                        or pipeline_job.status == "CancelRequested"
                        or pipeline_job.status == "Canceled"
                    ):
                        break
                else:
                    break

            if pipeline_job.status == "Completed" or pipeline_job.status == "Finished":
                logging.info("job completed")
            else:
                raise Exception("Sorry, exiting job with failure..")
    except Exception as ex:
        print(f"Exception raised in execute_pipeline {ex}")
        raise


def prepare_and_execute(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    cluster_name: str,
    cluster_size: str,
    cluster_region: str,
    min_instances: int,
    max_instances: int,
    idle_time_before_scale_down: int,
    env_base_image_name: str,
    conda_path: str,
    environment_name: str,
    env_description: str,
    wait_for_completion: bool,
    model_name: str,
    model_description: str,
    display_name: str,
    experiment_name: str,
    deploy_environment: str,
    build_reference: str,
    automl_model_name: str,
    automl_experiment_name: str,
    automl_compute_cluster_name: str,
    automl_cluster_size: str,
    automl_cluster_region: str,
    automl_min_instances: int,
    automl_max_instances: int,
    automl_idle_time_before_scale_down: int,
    output_file: Optional[str],
):
    """Prepare the pipeline and execute it.

    Checks all resource requirements for the pipleine and creates them if they do not exist. Then
    creates the pipeline and executes it.
    """
    compute = get_compute(
        subscription_id,
        resource_group_name,
        workspace_name,
        cluster_name,
        cluster_size,
        cluster_region,
        min_instances,
        max_instances,
        idle_time_before_scale_down,
    )

    automl_compute = get_compute(
        subscription_id,
        resource_group_name,
        workspace_name,
        automl_compute_cluster_name,
        automl_cluster_size,
        automl_cluster_region,
        automl_min_instances,
        automl_max_instances,
        automl_idle_time_before_scale_down,
    )

    environment = get_environment(
        subscription_id,
        resource_group_name,
        workspace_name,
        env_base_image_name,
        conda_path,
        environment_name,
        env_description,
    )
    print(f"Environment: {environment.name}, version: {environment.version}")

    pipeline_job = construct_pipeline(
        subscription_id,
        resource_group_name,
        workspace_name,
        compute.name,
        f"azureml:{environment.name}:{environment.version}",
        model_name,
        model_description,
        display_name,
        deploy_environment,
        build_reference,
        automl_model_name,
        automl_experiment_name,
        automl_compute.name,
    )

    execute_pipeline(
        subscription_id,
        resource_group_name,
        workspace_name,
        experiment_name,
        pipeline_job,
        wait_for_completion,
        output_file,
    )


def main():
    """Parse all args and execute the pipeline."""
    parser = argparse.ArgumentParser("build_environment")
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
        "--cluster_region",
        type=str,
        help="Azure Machine learning cluster region",
        default="eastus2",
    )
    parser.add_argument("--min_instances", type=int, default=0)
    parser.add_argument("--max_instances", type=int, default=4)
    parser.add_argument("--idle_time_before_scale_down", type=int, default=120)
    parser.add_argument(
        "--build_reference",
        type=str,
        help="Unique identifier for Azure DevOps pipeline run",
    )
    parser.add_argument(
        "--deploy_environment",
        type=str,
        help="execution and deployment environment. e.g. dev, prod, test",
    )
    parser.add_argument(
        "--experiment_name", type=str, help="Job execution experiment name"
    )
    parser.add_argument("--display_name", type=str, help="Job execution run name")
    parser.add_argument(
        "--wait_for_completion",
        type=bool,
        help="Set to True to wait for pipeline job completion",
    )
    parser.add_argument(
        "--environment_name",
        type=str,
        help="Azure Machine Learning Environment name for job execution",
        default="conda-based-devenv-py38-cpu",
    )
    parser.add_argument(
        "--env_base_image_name",
        type=str,
        help="Environment custom base image name",
        default="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        help="path to conda requirements file",
        default="model_factory/fridge_obj_det/mlops/environment/conda.yml",
    )
    parser.add_argument(
        "--env_description", type=str, default="Environment created using Conda."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="fridge-objects-automl-onnx",
        help="The name of the registered model.",
    )
    parser.add_argument(
        "--model-description",
        type=str,
        default="Best AutoML Object Detection ONNX model for fridge objects dataset.",
        help="The description of the registered model.",
    )
    parser.add_argument(
        "--automl_model_name",
        type=str,
        default="fasterrcnn_resnet18_fpn"
    )
    parser.add_argument(
        "--automl_experiment_name",
        type=str,
        default="automl-fridge-objects-detection-experiment"
    )
    parser.add_argument(
        "--automl_compute_cluster_name",
        type=str,
        help="The AML cluster name for running AutoML training experiments.",
        default="gpu-cluster-v100"
    )
    parser.add_argument(
        "--automl_cluster_size",
        type=str,
        help="AML cluster size for AutoML jobs.",
        default="STANDARD_NC6S_V3"
    )
    parser.add_argument(
        "--automl_cluster_region",
        type=str,
        help="AML cluster region for AutoML jobs.",
        default="eastus2",
    )
    parser.add_argument("--automl_cluster_min_instances", type=int, default=0)
    parser.add_argument("--automl_cluster_max_instances", type=int, default=4)
    parser.add_argument("--automl_cluster_idle_time_before_scale_down", type=int, default=120)
    parser.add_argument(
        "--output_file", type=str, required=False, help="A file to save run id"
    )

    args = parser.parse_args()

    prepare_and_execute(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.cluster_name,
        args.cluster_size,
        args.cluster_region,
        args.min_instances,
        args.max_instances,
        args.idle_time_before_scale_down,
        args.env_base_image_name,
        args.conda_path,
        args.environment_name,
        args.env_description,
        args.wait_for_completion,
        args.model_name,
        args.model_description,
        args.display_name,
        args.experiment_name,
        args.deploy_environment,
        args.build_reference,
        args.automl_model_name,
        args.automl_experiment_name,
        args.automl_compute_cluster_name,
        args.automl_cluster_size,
        args.automl_cluster_region,
        args.automl_cluster_min_instances,
        args.automl_cluster_max_instances,
        args.automl_cluster_idle_time_before_scale_down,
        args.output_file,
    )


if __name__ == "__main__":
    main()
