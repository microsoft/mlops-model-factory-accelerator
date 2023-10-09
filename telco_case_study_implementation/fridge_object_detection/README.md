# Sample implementation of Model factory and Usecase builder

## Context

This is a sample implementation of the model factory.

## Design and Architecture

- [Design of the Model factory](docs/01-model-factory-design.md)

## Adaptation from the accelerator and differences

### How this sample implementation is built on top of the accelerator

- Cloned the accelerator repository into a client specific repository.
- Configured the pipelines in Azure Devops, and made relevant changes to adapt to the client's ways of working.
- Extended the mlops runner to support the client's use-case.
- Inference code for the model was updated related to the specific model. 
  
### Differences in the sample implementation from the accelerator

- There are few changes to the folder structure in the sample implementation. The accelerator will have functional blocks (ml-ops, model, src) as top level folders. Where as in the sample implementation, we have the models at the top and the functional blocks will be within the specific models.
- In the accelerator, we have the devops pipeline to setup and trigger the `mlops-pipeline` in the common folder. While implementing, we had a scenario to pass in different variables for different models' ml-ops-pipeline, hence this is moved to the model specific folder.

## How to use this sample

- [Instructions to use this sample](docs/02-instructions.md)