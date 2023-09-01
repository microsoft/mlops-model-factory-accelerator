# High Level Conceptual Architecture

## Introduction

This document outlines the high level conceptual design for the sample use case builder and model factory. The use case builder and model factory are systems that automate the end-to-end process of developing, training, deploying, and managing machine learning models.

## Suggested Approach

![conceptual_architecture](/docs/assets/images/conceptual_architecture.png)

The above diagram depicts the high-level conceptual design of the sample solution.

Technically, a ML use case can be defined as:

*A combination of single or multiple machine learning models coupled together by an orchestration mechanism to address a specific business problem.*

The conceptual design has 3 main components which function together to generate and deliver use cases to the target edge devices.

The steps below depict the life cycle of a use case:

1. The first step in the journey of developing a new use case is to identify if the required Machine Learning models needed to fulfill the use case are already available in the Machine Learning Model factory or new ML models needs to be created.
2. Machine Learning Model Factory:
    - When the intended use case demands new ML models or updates to existing ones, the data scientist and ML engineers utilize the Machine Learning Model Factory to create, package, and handle the ML models at scale, while adhering to recommended MLOps practices.
    - The output from the Model Factory is one or more ML models packaged inside their own containerized environment (such as a Docker image).
3. Use Case Builder:
    - Once the dependent ML models are ready, the Use Case Builder is leveraged to build the orchestration layer, which performs pre-processing and post-processing steps for model inferencing.
    - The Use Case Builder also performs end-to-end validation of the use case by deploying the orchestration layer and the dependent ML models on a targeted edge device. The iterative process of building the use case continues until the validation passes.
    - Since multiple ML models are chained together to deliver a use case, its crucial to assess the end to end performance of the use case in a controlled environment before rolling it out at scale. The use case artifacts generated out of the Use Case Builder are deployed to the connected test edge device for carrying out this assessment manually: 
        - If the assessment fails, the dependent ML models are fine-tuned through the Model Factory and processed again through the Use Case Builder.
        - If the assessment is successful, the output from the Use Case Builder; which included the packaged orchestration layer and IoT edge deployment manifest, is published to a central repository for consumption. 
4. Use Case Deployment
    - The last leg in the journey of a use case is the deployment of the published use case to the targeted edge devices.
    - Leveraging the native IoT Edge capabilities, the use case deployment supports full and incremental deployments of the use case to the target edge devices.
