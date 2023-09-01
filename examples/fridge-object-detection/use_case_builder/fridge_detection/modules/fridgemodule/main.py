# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import json
import logging
import os
import signal
import sys
import time

import multiprocessing
from multiprocessing.context import DefaultContext

from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device import MethodResponse

from src.multiprocessing.controller import ProcessController
from src.frame_processor.inference_result_handler import E2ETestInferenceResultHandler

# Basic logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def initialize_iot_hub_communication(process_controller):
    client = IoTHubModuleClient.create_from_edge_environment()
    # client = IoTHubModuleClient.create_from_connection_string(os.environ["CONNECTION_STRING"])

    async def method_request_handler(method_request):
        """
        This is the method request handler from IoTHub
        """
        try:
            if str(method_request.name).lower() == "e2etest":
                print(
                    "Received direct message: {} {}\n".format(
                        str(method_request.name), str(method_request.payload)
                    )
                )

                json_request = json.loads(str(method_request.payload))
                e2e_test_video = json_request["test_video"]
                e2e_test_result = e2e_test(process_controller, e2e_test_video)

                # Create a method response indicating the method request was resolved
                resp_status = 200
                resp_payload = {"Response": e2e_test_result}
                method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
                await client.send_method_response(method_response)
            else:
                print("Invalid method name. \
                    acceptable methods include 'E2ETest'.")
                method_response = MethodResponse.create_from_method_request(
                    method_request, 400,
                    "{\"results\": \"Invalid method name. \
                        acceptable methods include \'E2ETest\'. \"}"
                )
                await client.send_method_response(method_response)

        except Exception as e:
            print(e)
            method_response = MethodResponse.create_from_method_request(
                method_request, 400, "{\"results\": \"fail\"}"
            )
            await client.send_method_response(method_response)

    try:
        # Set handler on the client
        client.on_method_request_received = method_request_handler
    except Exception:
        # Cleanup if failure occurs
        client.shutdown()
        raise


def main(multiprocessing_context: DefaultContext):
    """
    Main entry point for the orchestrator module.
    """
    logger = logging.getLogger(__name__)

    # Start orchestrator processes
    process_controller = ProcessController(multiprocessing_context)
    try:
        process_controller.start()
    except Exception:
        logger.exception("Error starting process controller")
        sys.exit(1)

    # Initialize IoT Hub communication
    initialize_iot_hub_communication(process_controller)

    # Main thread sleeps to keep the program running
    logger.info("Running main process in the background...")
    while True:
        time.sleep(100000000)


def e2e_test(process_controller, e2e_test_video):
    process_controller.stop()
    logger.info("Starting E2E test...")
    process_controller.start(video_source=e2e_test_video, inference_result_handler=E2ETestInferenceResultHandler)
    logger.info("Waiting for inference result...")
    inference_result = process_controller.inference_result_handler.get_inference_result()
    process_controller.restart()
    return inference_result


def exit_gracefully(*args):
    logger = logging.getLogger(__name__)
    logger.warning("Received SIGINT/SIGTERM signal. Exiting gracefully...")
    sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    multiprocessing_context = multiprocessing.get_context()
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    main(multiprocessing_context)
