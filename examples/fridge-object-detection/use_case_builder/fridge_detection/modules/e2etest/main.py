# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import asyncio
import logging
import sys
import signal
import threading
from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device import MethodResponse
import os
import requests
import datetime
import json

# Basic logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Event indicating client stop
stop_event = threading.Event()


def validate_e2e_result(response):
    result = "Fail"  # Initialize the result to "Fail"
    if response is not None:
        if response['status'] == 200:
            inference_result = response['payload']['Response']
            expected_result = {'carton': 2, 'milk_bottle': 1, 'can': 1, 'water_bottle': 1}
            if inference_result == expected_result:
                result = "Success"  # Update the result to "Success"
    return result


def create_client():
    client = IoTHubModuleClient.create_from_edge_environment()

    async def method_request_handler(method_request):
        """
        This is the method request handler from IoTHub
        """
        try:
            if (str(method_request.name)).lower() == "healthcheck":
                logger.info(f"Received direct method request: {str(method_request.name)}")

                url = "http://inferencemodule:8080/healthcheck"
                health_check_response = requests.get(url=url, timeout=5)
                logger.info(f"Health check response from inference module: {health_check_response.text}")

                if health_check_response.status_code == 200:
                    method_response = MethodResponse.create_from_method_request(
                        method_request, 200, {"status": "OK"}
                    )
                    await client.send_method_response(method_response)
                else:
                    method_response = MethodResponse.create_from_method_request(
                        method_request, 500, {"status": "FAIL"}
                    )
                    await client.send_method_response(method_response)
            elif (str(method_request.name)).lower() == "e2etesttrigger":
                logger.info(f"Received direct method request: {str(method_request.name)}")

                device_id = os.environ["IOTEDGE_DEVICEID"]

                payload = {"test_video": "rtsp://rtsp_sim:554/media/fridge-object-detection-e2e-test.mkv"}
                json_payload = json.dumps(payload)

                logger.info("Triggering E2E test...")
                logger.info(f"Sending direct method request to orchestrator with payload: {json_payload}")

                response = await client.invoke_method(
                    method_params={
                        "methodName": "E2ETest",
                        "payload": json_payload,
                        "responseTimeoutInSeconds": 120
                    },
                    device_id=device_id,
                    module_id="fridgemodule")
                logger.info(f"Received response from orchestrator: {response}")
                logger.info("Validating E2E test result...")
                result = validate_e2e_result(response)
                logger.info(f"Status of E2E Test: {result}")
                resp_status = 200
                resp_payload = {"test_result": result}
                method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
                await client.send_method_response(method_response)
            else:
                print("Invalid method name. \
                    acceptable methods include 'e2etesttrigger'.")
                method_response = MethodResponse.create_from_method_request(
                    method_request, 400,
                    "{\"results\": \"Invalid method name. \
                        acceptable methods include \'e2etesttrigger\'. \"}"
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
    except:
        # Cleanup if failure occurs
        client.shutdown()
        raise

    return client


async def run_e2e_test(client):
    # Customize this coroutine to do whatever tasks the module initiates
    # e.g. sending messages
    while True:
        await asyncio.sleep(1000)


def main():
    print("IoT Hub Client for Python")
    print(datetime.datetime.now())
    # NOTE: Client is implicitly connected due to the handler being set on it
    client = create_client()

    # Define a handler to cleanup when module is is terminated by Edge
    def module_termination_handler(signal, frame):
        print("IoTHubClient sample stopped by Edge")
        stop_event.set()

    # Set the Edge termination handler
    signal.signal(signal.SIGTERM, module_termination_handler)

    # Run the sample
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_e2e_test(client))
    except Exception as e:
        print("Unexpected error %s " % e)
        raise
    finally:
        print("Shutting down IoT Hub Client...")
        loop.run_until_complete(client.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
