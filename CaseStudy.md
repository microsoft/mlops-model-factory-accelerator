# Case Study – “Accelerating ML Model Development and Deployment on the edge using MLOps Model Factory Accelerator”


## Purpose
This document depicts a real-world case study on how implementing a model factory enabling a large Telco customer, to revolutionize its services by addressing challenges related to speed and agility in building and deploying machine learning models at scale.


## Customer Overview
A large Telco company has expanded its services to B2B customers with a data-driven analytics product focused on tracking the movement of objects and their presence in various environments. This solution utilizes cameras, edge computing, artificial intelligence, machine learning, and data analytics to provide valuable insights into the utilization of open spaces, infrastructure. These insights facilitate cost-effective and real-time planning decisions. 

The company aims to broaden its product's applicability across industries, offering services like Computer Vision as a Service, leveraging their telco infrastructure while ensuring compatibility with low-bandwidth scenarios. Their target markets range from cross industries to novel applications, necessitating rapid deployment and scalable automation for success.


## Problem Statement
The existing product has limitations making it challenging to scale the solution and quickly support new scenarios and customers.  

![image](/docs/images/ProblemStatement_FishBone.png)

<ins> Problem Areas Associated with the Broader Problem Statement: </ins>
- <ins>A)	Model Generation:</ins> The extended turnaround time required for creation and updates of machine learning models hinders the ability to onboard new use cases or updating the existing ones. Absence of robust performance evaluation mechanisms for models adds complexity, making it difficult to assess and improve model effectiveness.
- <ins>B)	Deployment Process:</ins> The current deployment process poses challenges, including semi-automated deployments with heavy Docker image sizes significantly slowing down the deployment process. Additionally, the tightly coupled model and application code hinders the agility required for swift updates and modifications.
- <ins>C)	Deployment Pace:</ins> The inefficient deployment process leads to extended deployment timelines that can span several weeks. The lack of support for limited network bandwidth frequently increases these delays, causing deployments to become disproportionately time-consuming.
- <ins>D)	Data Collection:</ins> Existing manual process for  data collection from the Edge devices is error prone and often leads to scalability issues.
- <ins>E)	Cost:</ins> Higher network costs contributes to increased operational costs.
  
<ins> Few Stats on the Existing Architecture:</ins>

|   Time taken for deploying multiple scenarios(5-10) against model      |     ~ Multiple Weeks                         |
|------------------------------------------------------------------------|----------------------------------------------|
| Model Package Container Size                                           | * 10GB for X86(AMD Build) <br/> * 7GB for ARM  



## Solution Overview
To address these key challenges, the existing architecture has been revamped & reimagined by decoupling the model generation, from the business use case development and deployment process. This newer approach focuses on building a framework which can generate end-to-end machine learning use case with speed and agility.
Technically, a ML use case can be defined as: A combination of single or multiple machine learning models coupled together by an orchestration mechanism to address a specific business problem.

The conceptual design has 3 key components which function together to generate and deliver use cases to the target edge devices.

![image](/docs/images/ConceptualDesign.png)

- #### <ins>Machine Learning Model Factory:</ins>
  - The ML Model factory address the challenges around Model Generation by streamlining the process of generating and managing machine learning models. It enables the data scientist and ML engineers collaborate to develop, package, and manage the ML models at scale, while following the standard MLOps practices. 
  - The output of the Model Factory is one or more ML models packaged inside their own containerized environment (such as a Docker image).
- #### <ins> Use Case Builder:</ins>
   - The Use Case Builder provides a solid foundation for the Deployment processes and Pace by providing a convention based framework which enables the developers to build the orchestration layer responsible for interfacing with the underlying edge devices, peripherals and interacting with the ML Model containers for inferencing.
   - It also enables other key features like end-to-end automated deployment validation of the use case on the targeted edge devices, packaging and publishing the validated orchestration code and edge deployment manifest to a central repository.
- #### <ins>Use Case Deployment:</ins>
   - Use case deployment enables rolling full and incremental deployments of the published use case to the edge devices leveraging the native IoT Edge capabilities.
   - **Due to the customer's well-established maturity in edge deployment processes, this component will be developed by the customer and would seamlessly integrate with the Use Case builder to provide an end-to-end deployment experience.


## Technical Implementation
The following section provided an high-level overview of the key solution components.
### Model Factory Design
Model factory enables developing and publishing ML models at scale by combining MLOps and DevOps pipelines with a standardized convention based approach for creating an efficient and scalable system which allows generating, training, and maintaining ML models.

MLOps pipelines built on top of Azure Machine Learning at its core are responsible for automating and orchestrating various stages of a ML model lifecycle ranging from data pre-processing, feature engineering, model training, hyperparameter tuning, and model evaluation.

DevOps pipelines built on top of Azure DevOps are responsible for packaging the trained ML models into Docker containers and facilitating their deployment across multiple environments.

The workflow of a model from a developer environment to a production container registry is depicted below. This also depicts various validation checks that are enforced in various stages of the workflow. 
 ![image](/docs/images/MLModelFactory_Design.png)

#### Key Features
- Supports generation of multiple ML Models
- MLOps pipeline for Data preparation, transformation, Model Training, evaluation, scoring and registration.
- Each ML Model is packaged in an independent Docker Image
- Model verification before storing the Docker image
- All Docker images are stored in Azure Container Registry
- Builds and deploys Smoke Test module on Edge device

### Use Case Builder Design 
Use case builder provides a convention based approach to enable the developers to build and manage multiple use-cases across different environments. The framework provides the ability to create a single deployment manifest that bundles one or more ML models along the orchestration layer to server an end to end use case.

At its core, the Use Case Builder leverages Azure IoT Edge and Azure DevOps pipelines to allow the developers build and integration test the orchestration layer in conjunction with the dependent ML Models on the targeted edge devices.

The workflow of Use Case Builder from a developer environment to a production container registry is depicted below .
  ![image](/docs/images/UseCaseBuilder_Design.png)
#### Key Features
- Bundles model reference and orchestration layer into a single deployable entity based on the business scenario.
- DevOps pipeline templates for the use-cases to build, test and publish artifacts.
- Ability to author and manage the IoT Edge manifest for a use-case specific deployment on the targeted edge devices.
- Validation of the complete use-cases using an integrated end-to-end test running on the actual edge device.
- Provision to scaffold a use-case with a predefined folder structure.
### Data Collection on the Edge Design
Data Collection on the Edge component enables the customer to have a secure and automated process for data collection (images) from the edge devices, and use those images as a source for subsequent model training (post annotation).
This component leverages [Azure Blob Storage IoT Edge](https://learn.microsoft.com/en-us/azure/iot-edge/how-to-store-data-blob?view=iotedge-1.4) modules at its core to provides a low-code mechanism for the edge devices to push binary data to remote azure blob storage accounts.
 
 ![image](/docs/images/DataCollection_Design.png)

#### Key Features
- A no-code low-code solution which can be quickly setup to automate data collection on edge.
- Supports intermittent internet connectivity and provides an asynchronous way for data upload from edge.
- Support upload of high-quality large size images for model training on the cloud

### Deployment Size Optimizations for Constrained Network
To address key challenges around heavy edge images leading to exponential deployments (caused by multiple failures), lack of support for limited network bandwidth and high network cost, a two-stage approach was implemented.
#### <ins>1.	Orchestrated Inferencing :</ins> The concept of Orchestrated inferencing was introduced which dictate the design based on decoupling the ML models and orchestration logic into separate docker containers. 

 ![image](/docs/images/Orchestrated_Inferencing.png)

This decouple design resulted into multiple benefits:- 
-	Support for selective ML model deployment instead of bundling all ML models and orchestration logic into a single bulky image.
-	Ability to incrementally update ML models and Orchestration code individually thereby reducing the impact radius.
-	Reduction of the overall n/w footprint due lighter edge deployments thereby reducing operational cost.
#### <ins>2.	Docker image optimizations:</ins> To further reduce the size of individual docker containers, [docker layers optimizations](https://docs.docker.com/build/cache) were implemented to reduces the size of the image and the amount of data that needed to be transferred over the wire. 
### Deployment Size Reduction Data 
The combination of the aforementioned approaches lead to a drastic reduction of the deployment size. The table below depicts this data for a reference use case.

|     Deployment Size                      |                           |                                                                  |  
|------------------------------------------|---------------------------|------------------------------------------------------------------|
|                                          |     Existing Approach     |     Orchestrated   Inferencing + Docker Layer   Optimizations    |  
|     First   time load                    |     9 GB                  |     8 GB                                                         | 
|     Delta load – Model + code changes    |     4 GB                  |     ~100 MB                                                      | 
|     Delta   load – Only code changes     |     4 GB                  |     ~30 KB                                                       |   

## Conclusion

In conclusion, this technical case study highlights the significant challenges faced by the customer across various facets of ml model deployment and operations. These encompassed obstacles from slow model generation to challenging deployment processes and high operational cost.
The solution implemented for the customer has not only streamlined their ML model generation and deployment processes but have also ushered in a structured, scalable approach to machine learning operations at scale. This meets the customer's objective of building ML models with speed and agility while simultaneously enhancing cost-effectiveness and reducing operational complexities.

## References

-	[MLOps Model Factory Accelerator](https://github.com/microsoft/mlops-model-factory-accelerator)
-	[MLOps Accelerator for Edge](https://github.com/microsoft/mlops-accelerator-for-edge)






