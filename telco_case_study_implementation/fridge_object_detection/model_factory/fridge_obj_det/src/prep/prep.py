"""Prep the fridge objects dataset into train, val, test registered AML MLTables."""
import argparse
import os
import urllib
from zipfile import ZipFile
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from fridge_obj_det.src.prep.voc_jsonl_converter import (
    VOCJSONLConverter,
    write_json_lines,
)
from common.mlops.get_aml_client import get_aml_client
import random
import logging


def get_fridge_objects_dataset(dataset_parent_dir: os.path) -> str:
    """Get the open fridge objects dataset.

    Download the fridge objects dataset, unzip it, remove the zip file
    and return the dataset directory where the data is stored.

    Args:
        dataset_parent_dir (str): parent directory where the dataset will be stored

    Returns:
        str: dataset directory where the data is stored
    """
    try:
        # download data
        download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"

        # Extract current dataset name from dataset url
        dataset_name = os.path.split(download_url)[-1].split(".")[0]
        # Get dataset path for later use
        dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

        # Get the data zip file path
        data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

        # Download the dataset
        if not os.path.exists(dataset_dir):
            urllib.request.urlretrieve(download_url, filename=data_file)

            # extract files
            with ZipFile(data_file, "r") as zip:
                print("extracting files...")
                zip.extractall(path=dataset_parent_dir)
                print("done")
            # delete zip file
            os.remove(data_file)

        return dataset_dir
    except Exception as ex:
        print(f"Exception in get_fridge_objects_dataset: {ex}")
        raise


def create_ml_table_file(filename: str) -> str:
    """Create ML Table definition.

    Args:
        filename (str): JSONL file name that will be
         used as the ML Table source

    Returns:
        str: ML Table definition
    """
    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def save_ml_table_file(output_path: str, mltable_file_contents: str) -> None:
    """Save ML Table definition to output_path/MLTable.

    Args:
        output_path (str): path to save the ML Table definition
        mltable_file_contents (str): ML Table definition
    """
    try:
        print(f"Saving MLTable file {output_path}")
        with open(os.path.join(output_path, "MLTable"), "w") as ml_table_file:
            ml_table_file.write(mltable_file_contents)
    except Exception as ex:
        print(f"Exception while saving MLTable file: {ex}")
        raise


def main(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    fridge_objects_uri_folder_name: str,
    train_mltable_name: str,
    val_mltable_name: str,
    test_mltable_name: str,
    train_mltable: str,
    val_mltable: str,
    test_mltable: str,
):
    """Split the fridge objects dataset into train, val, test and register as MLTables.

    Args:
        client_id (str): AAD client ID.
        client_secret (str): AAD client secret.
        tenant_id (str): AAD tenant ID.
        fridge_objects_uri_folder_name (str): fridge objects images URI folder name in AML blob storage.
        train_mltable_name (str): train split registered MLTable name in AML.
        val_mltable_name (str): val split registered MLTable name in AML.
        test_mltable_name (str): test split registered MLTable name in AML.
    """
    # Create ML Client
    try:
        client = get_aml_client(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        # TODO: check that URI folder is not already registred before doing all this
        # prep, can also have a parameter to force an update if required and the dataset
        # already exists

        # Download fridge objects open dataset
        dataset_parent_dir = "./data"
        os.makedirs(dataset_parent_dir, exist_ok=True)
        dataset_dir = get_fridge_objects_dataset(dataset_parent_dir)

        print(f"Dataset downloaded to {dataset_dir}")

        # Upload to AML workspace as a URI folder data asset
        fridge_objects_data = Data(
            path=dataset_dir,
            type=AssetTypes.URI_FOLDER,
            description="Fridge-items images Object detection",
            name=fridge_objects_uri_folder_name,
        )
        uri_folder_data_asset = client.data.create_or_update(fridge_objects_data)

        print(f"URI folder data asset created {str(uri_folder_data_asset)}")

        # Convert the downloaded fridge objects dataset that is annotated in VOC format to JSONL
        # The JSONL will reference the images in the URI folder data asset in AML
        base_url = os.path.join(uri_folder_data_asset.path, "images/")
        converter = VOCJSONLConverter(
            base_url=base_url, xml_dir=os.path.join(dataset_dir, "annotations")
        )
        jsonl_annotations = os.path.join(dataset_dir, "annotations_voc.jsonl")
        write_json_lines(converter, jsonl_annotations)

        training_mltable_path = train_mltable
        validation_mltable_path = val_mltable
        test_mltable_path = test_mltable

        # First, let's create the folders if they don't exist
        os.makedirs(training_mltable_path, exist_ok=True)
        os.makedirs(validation_mltable_path, exist_ok=True)
        os.makedirs(test_mltable_path, exist_ok=True)

        print(f"Trainging mltable path: {training_mltable_path}")
        print(f"Validation mltable path: {validation_mltable_path}")
        print(f"Test mltable path: {test_mltable_path}")

        # Path to the training, validation, test JSONL files
        train_annotations_file = os.path.join(
            training_mltable_path, "train_annotations.jsonl"
        )
        validation_annotations_file = os.path.join(
            validation_mltable_path, "validation_annotations.jsonl"
        )
        test_annotations_file = os.path.join(
            test_mltable_path, "test_annotations.jsonl"
        )

        with open(jsonl_annotations, "r") as annot_f:
            json_lines = annot_f.readlines()

        random.shuffle(json_lines)
        train_annotations = json_lines[: int(len(json_lines) * 0.6)]
        val_annotations = json_lines[
            int(len(json_lines) * 0.6): int(len(json_lines) * 0.8)
        ]
        test_annotations = json_lines[int(len(json_lines) * 0.8):]
        assert len(train_annotations) + len(val_annotations) + len(
            test_annotations
        ) == len(json_lines)

        with open(train_annotations_file, "w") as train_f:
            for json_line in train_annotations:
                train_f.write(json_line)

        with open(validation_annotations_file, "w") as validation_f:
            for json_line in val_annotations:
                validation_f.write(json_line)

        with open(test_annotations_file, "w") as test_f:
            for json_line in test_annotations:
                test_f.write(json_line)

        # Create and save train, val, test mltables
        train_mltable_file_contents = create_ml_table_file(
            os.path.basename(train_annotations_file)
        )
        save_ml_table_file(training_mltable_path, train_mltable_file_contents)

        validation_mltable_file_contents = create_ml_table_file(
            os.path.basename(validation_annotations_file)
        )
        save_ml_table_file(validation_mltable_path, validation_mltable_file_contents)

        test_mltable_file_contents = create_ml_table_file(
            os.path.basename(test_annotations_file)
        )
        save_ml_table_file(test_mltable_path, test_mltable_file_contents)

        print(f"mltable name {train_mltable_name}")
        print(f"val_mltable_name name {val_mltable_name}")
        print(f"test_mltable_name name {test_mltable_name}")
        # Register train/val/test MLTable folders as data assets in AML
        train_data_asset = Data(
            path=training_mltable_path,
            type=AssetTypes.MLTABLE,
            description="Training split (60%) for fridge objects dataset.",
            name=train_mltable_name,
        )
        client.data.create_or_update(train_data_asset)

        val_data_asset = Data(
            path=validation_mltable_path,
            type=AssetTypes.MLTABLE,
            description="Validation split (20%) for fridge objects dataset.",
            name=val_mltable_name,
        )
        client.data.create_or_update(val_data_asset)

        test_data_asset = Data(
            path=test_mltable,
            type=AssetTypes.MLTABLE,
            description="Test split (20%) for fridge objects dataset.",
            name=test_mltable_name,
        )
        client.data.create_or_update(test_data_asset)
    except Exception as ex:
        print(f"Exception in main {ex}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_id", type=str, help="Azure client id")
    parser.add_argument("--client_secret", type=str, help="Azure client secret")
    parser.add_argument("--tenant_id", type=str, help="Azure tenant id")

    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument(
        "--resource_group_name", type=str, help="Azure Machine learning resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, help="Azure Machine learning Workspace name"
    )
    parser.add_argument(
        "--fridge_objects_uri_folder_name",
        type=str,
        help="URI folder name for the fridge items images dataset.",
        default="fridge-items-images-object-detection",
    )
    parser.add_argument(
        "--train_mltable_name",
        type=str,
        help="MLTable name for the training split.",
        default="fridge-objects-train-mltable",
    )
    parser.add_argument(
        "--val_mltable_name",
        type=str,
        help="MLTable name for the validation split.",
        default="fridge-objects-val-mltable",
    )
    parser.add_argument(
        "--test_mltable_name",
        type=str,
        help="MLTable name for the test split.",
        default="fridge-objects-test-mltable",
    )

    parser.add_argument(
        "--train_mltable",
        type=str,
        help="MLTable for the training split.",
    )
    parser.add_argument(
        "--val_mltable",
        type=str,
        help="MLTable for the validation split.",
    )
    parser.add_argument(
        "--test_mltable",
        type=str,
        help="MLTable for the test split.",
    )

    args = parser.parse_args()
    logging.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    main(
        args.client_id,
        args.client_secret,
        args.tenant_id,
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.fridge_objects_uri_folder_name,
        args.train_mltable_name,
        args.val_mltable_name,
        args.test_mltable_name,
        args.train_mltable,
        args.val_mltable,
        args.test_mltable,
    )
