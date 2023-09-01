import asyncio
import datetime

import signal
import sys
import threading

from azure.iot.device import MethodResponse
from azure.iot.device.aio import IoTHubModuleClient

from healthcheck import Healthcheck
from model_factory.common.smoke_test.modules.TestModule.response import Response
from model_factory.common.smoke_test.modules.TestModule.test_runner import TestRunner


# Event indicating client stop
stop_event = threading.Event()


def create_client():
    """
    Direct Method Client to do smoke test using http request
    """
    client = IoTHubModuleClient.create_from_edge_environment()

    async def _send_response_as_bad_request(method_request):
        default_response = Response(400, '{"results": "Invalid method name. \
                            accepctable methods include \'smokeTest\'. "}')
        await _send_response(method_request, default_response)

    async def _send_response_as_internal_error(method_request):
        default_response = Response(500, '{"results": "Something went wrong. \
            check the exception logs"}')
        await _send_response(method_request, default_response)

    async def _send_response(method_request, response: Response):
        method_response = MethodResponse.create_from_method_request(
                        method_request, response.status, response.payload
            )
        await client.send_method_response(method_response)
        await client.send_message_to_output("done", "output1")

    async def method_request_handler(method_request):
        """
        This is the method request handler from IoTHub
        """
        try:
            await client.send_message(
                "Received Method Request: " + str(method_request.name)
            )
            print(
                "Received direct message: {} {}\n".format(
                    str(method_request.name), str(method_request.payload)
                )
            )
            if str(method_request.name) == "healthcheck":
                port = method_request.payload["port"]
                print(f"Health check: port: {port}")
                response = Healthcheck().execute(port)
                await _send_response(method_request, response)
            elif str(method_request.name) == "smokeTest":
                port = method_request.payload["port"]
                model_type = method_request.payload["model_type"]
                print(f"Smoke test: port: {port}, model_type: {model_type}")
                response = TestRunner().execute_smoke_test(port, model_type)
                await _send_response(method_request, response)
            else:
                await _send_response_as_bad_request(method_request)

        except Exception as ex:
            print(f"Exception in method_request_handler: {ex}")
            await _send_response_as_internal_error(method_request)

    try:
        # Set handler on the client
        client.on_method_request_received = method_request_handler
    except Exception:
        client.shutdown()
        raise

    return client


async def run_test(client):
    """
    Runner
    """
    while True:
        await asyncio.sleep(1000)


def main():
    """
    Main method
    """
    if not sys.version >= "3.5.3":
        raise Exception(
            "The sample requires python 3.5.3+. Current version of Python: %s"
            % sys.version
        )
    print("IoT Hub Client for Python")
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
        loop.run_until_complete(run_test(client))
    except Exception as ex:
        print("Unexpected error %s " % ex)
        raise
    finally:
        print("Shutting down IoT Hub Client...")
        loop.run_until_complete(client.shutdown())
        loop.close()


if __name__ == "__main__":
    
    main()
