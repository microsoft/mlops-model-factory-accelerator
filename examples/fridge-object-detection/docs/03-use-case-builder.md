# Usecase Builder

This document contains the design for the usecase builder.

## Introduction

A “Use case” refers to a business problem/scenario that needs to be deployed as a single unit of work. Example. Vehicle detection, Cart tracking.

Use case builder provides convention for users to build and manage multiple use-cases across different environments. Use case builder provides the ability to create a single deployment manifest that bundles one or more models along with their pre-processing and post-processing logic.


## Features of Use case builder

- Bundles model and orchestration into a single deployable entity based on the business scenario.
- DevOps pipeline templates for the use-cases to build, test and publish.
- Ability to author and manage the IoT manifest for the use-case deployment.
- Validation of use-cases using an integrated end-to-end test running on an edge device.
- Provision to scaffold a use-case with a predefined folder structure.

## Development workflow for Use-case builder

While developing a new use case, a feature branch is forked from the main branch. User(s) working on this feature will fork out branches from the feature branch and will constantly integrating their code into this feature branch. The PR validation checks between the user branch and the feature branch is kept lean to enable frequent merging of code from the user branch.

The feature branch is protected by branch policies to enable only PR based merges. Post merging the code, feature branch runs all the validations and ensures the code within the branch is in sanity.

While merging the feature branch into main, again all the validations (unit testing, e2e testing) will happen to ensure only quality code reaches the main. Main branch will act as a release branch as well, and the final use-case containers and manifest are released from main branch on all PR merges into the main branch.

![development flow of use case builder](/docs/assets/images/usecase_builder_branch_strategy.png)

## Manifest file management

- Deployment manifest files are manually authored and managed within the use-case
- Manifest will contain all the edge modules that are used by the use-case, along with the routes
