# Overview

This document contains the instructions to use this sample.

## Prerequisite

Following azure resources are required to run this sample:

1. Azure AML Workspace:
    AML workspace also create following resources:
    1. Application Insights
    1. Azure Keyvault
    1. Azure Blob Storage
1. Azure Container Registry
1. Azure IotHub
1. Azure VM: It will be used for smoke test.
1. Azure VM: It will be used for e2e test.
1. Azure Devops
1. Service Principal

## Steps

1. Provision Azure AML workspace.
1. Provision Azure Container registry.
1. Provision Azure IoTHub.
1. Provision two Azure VMs using [this ARM template](https://github.com/Azure/iotedge-vm-deploy), so that they be used IoTEdge Device.
1. Create Service Connection in Azure Devops providing access to resource group.
1. Create two variable groups in Azure Devops
    1. mlops_platform_dev_vg
    1. mlops_platform_prod_vg  
    Add following variables to the both these variable group. Here different ACR's can be used for prod and dev env.
    1. ACR_URL : Azure container registry url
    1. ACR_USERNAME : Azure container registry username
    1. ACR_PASSWORD: Azure container registry password
    1. AZURE_RM_SVC_CONNECTION: Service Connection name
    1. KEYVAULT_NAME: Keyvault name
1. Create Service Principal  
`az ad sp create-for-rbac --name <name> --role owner --scopes /subscriptions/<subid>/resourceGroups/<resorucegroup>`
1. Give Service Principal access to AML workspace and Keyvault
1. Give Service Connection access to AML worksapce and keyvault.
1. Add following secrets to Azure Keyvault:
    1. aml-service-principal-id: Service Principal created in step 7
    1. aml-service-principal-secret: Service Principal secret
    1. tenant-id
    1. applicationinsights-connection-string: Application insights connection string

    Following variables are related to ACR.
    1. registry-uri: Dev ACR url
    1. registry-password : Dev ACR password
    1. registry-username: Dev ACR username  
    Dev ACR
    1. registry-uri-dev: Dev ACR url
    1. registry-username-dev: Dev ACR username
    1. registry-password-dev: Dev ACR password  
    Prod ACR
    1. registry-uri-prod: Prod ACR url
    1. registry-username-prod: Prod ACR username
    1. registry-password-prod: Prod ACR password
1. Update model_config(`model_factory\fridge_obj_det\config\model_config.json`) with required values.
1. Update use_case_config.json(`use_case_builder\fridge_detection\use_case_config.json`) with required values.
1. Create Azure Pipelines using following yaml files.
    1. Model factory Pipelines:
        1. fridge_obj_det_dev_pipeline: `model_factory\fridge_obj_det\devops\pipelines\fridge_obj_det_dev_pipeline.yml`
        1. fridge_obj_det_main_pipeline:
    `model_factory\fridge_obj_det\devops\pipelines\fridge_obj_det_main_pipeline.yml`
    1. Use case builder pipelines:
        1. fridge_dev_pipeline:
    `use_case_builder\fridge_detection\devops\pipelines\fridge_dev_pipeline.yml`
        1. fridge_main_ci_pipeline:
    `use_case_builder\fridge_detection\devops\pipelines\fridge_main_ci_pipeline.yml`
        1. fridge_main_pr_pipeline:
    `use_case_builder\fridge_detection\devops\pipelines\fridge_main_pr_pipeline.yml`
    1. RTSP builder pipeline:
       1. rtsp_builder_pipeline: `use_case_builder\common\devops\pipelines\pipeline_rtsp_builder.yml`

1. Execution of pipelines
    1. Model Factory Pipelines:  
       - Model factory dev pipeline executes AML pipelines and creates model container docker image and pushes it to ACR.  
       - Model factory main pipeline executes AML pipeline, creates docker image, performs smoke test on edge device and pushes image to ACR.
    2. Use case builder pipelines:
       - Use case builder dev pipeline perform validations, builds docker image for the orchestration container, pushes it to ACR, performs deployment to the e2e test device and runs e2e test.
       - Use case builder main ci pipeline will build docker image, and creates a deployment manifest for the deployment.
    3. RTSP builder pipeline:
       - This sample uses RTSP Simulator for e2etest. This pipeline will build RTSP simulator docker image and pushes it to ACR.
