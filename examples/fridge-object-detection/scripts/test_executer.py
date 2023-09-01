import asyncio
import json
import time
import sys

TEST_RESULT_LIST = []


async def health_check(device_name: str, test_result_list: list, ):
    health_check_command = "az iot hub invoke-module-method --method-name healthcheck -n {IOTHUB_NAME_PARAM} -d {DEVICE_PARAM} -m {TEST_DOCKER_NAME_PARAM} --method-payload '{{\"port\":\"{PORT_NUMBER_PARAM}\",\"model_type\":\"{MODEL_TYPE_PARAM}\"}}'".format(
        IOTHUB_NAME_PARAM=IOTHUB_NAME, DEVICE_PARAM=device_name, TEST_DOCKER_NAME_PARAM=TEST_DOCKER_NAME, PORT_NUMBER_PARAM=PORT_NUMBER, MODEL_TYPE_PARAM=MODEL_TYPE)
    duration = int(DURATION)  # Total duration in minutes
    interval = int(INTERVAL)  # Check interval in minutes
    interval_in_seconds = interval * 60
    end_time = time.time() + duration * 60
    healthcheck_status = False
    while time.time() < end_time:
        completed_healthcheck_process = await asyncio.create_subprocess_shell(health_check_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await completed_healthcheck_process.communicate()
        if stdout:
            response = json.loads(stdout.decode())
            if response["status"] == 200:
                healthcheck_status = True
                break
        time.sleep(interval_in_seconds)
    if not healthcheck_status:
        test_result_list.append({"device": "{device_name_param}".format(
            device_name_param=device_name), "status": "fail", "reason": "health_check_failed"})
        return False
    return True

async def smoke_test(device_name: str, test_result_list: list):
    smoke_test_command = "az iot hub invoke-module-method -n {IOTHUB_NAME_PARAM} -d {DEVICE_PARAM} -m {TEST_DOCKER_NAME_PARAM} --method-name 'smokeTest' --method-payload '{{\"port\":\"{PORT_NUMBER_PARAM}\",\"model_type\":\"{MODEL_TYPE_PARAM}\"}}'".format(
        IOTHUB_NAME_PARAM=IOTHUB_NAME, DEVICE_PARAM=device_name, TEST_DOCKER_NAME_PARAM=TEST_DOCKER_NAME, PORT_NUMBER_PARAM=PORT_NUMBER, MODEL_TYPE_PARAM=MODEL_TYPE)
    completed_e2e_process = await asyncio.create_subprocess_shell(smoke_test_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await completed_e2e_process.communicate()
    if stdout:
        response = json.loads(stdout.decode())
        if response["status"] == 200:
            test_result_list.append({"device": "{device_name_param}".format(
                device_name_param=device_name), "status": "success", "reason": "smoke_test_passed"})
            return
    test_result_list.append({"device": "{device_name_param}".format(
        device_name_param=device_name), "status": "fail", "reason": "smoke_test_failed"})
    return


async def e2e_test(device_name: str, test_result_list: list):
    e2e_command = "az iot hub invoke-module-method -n {IOTHUB_NAME_PARAM} -d {DEVICE_PARAM} -m {TEST_DOCKER_NAME_PARAM} --method-name e2etesttrigger".format(
        IOTHUB_NAME_PARAM=IOTHUB_NAME, DEVICE_PARAM=device_name, TEST_DOCKER_NAME_PARAM=TEST_DOCKER_NAME)
    print(e2e_command)
    completed_e2e_process = await asyncio.create_subprocess_shell(e2e_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await completed_e2e_process.communicate()
    if stdout:
        response = json.loads(stdout.decode())
        if response["status"] == 200 and response["payload"]["test_result"].upper() == "SUCCESS":
            test_result_list.append({"device": "{device_name_param}".format(
                device_name_param=device_name), "status": "success", "reason": "e2e_test_passed"})
            return
    test_result_list.append({"device": "{device_name_param}".format(
        device_name_param=device_name), "status": "fail", "reason": "e2e_test_failed"})
    return


async def run_smoke_test(device_name: str, test_result_list: list):
    print("Running smoke test...")
    if await health_check(device_name, test_result_list, ):
        await smoke_test(device_name, test_result_list, )


async def run_e2e_test(device_name: str, test_result_list: list):
    print("Running E2E test...")
    if await health_check(device_name, test_result_list, ):
        await e2e_test(device_name, test_result_list, )


async def main():
    try:
        if len(DEVICE_LIST) != 0:
            if RUN_TYPE.upper() == "SMOKE_TEST":
                coroutines = [run_smoke_test(device, TEST_RESULT_LIST, ) for device in DEVICE_LIST]
                # Gather and execute the coroutines
                await asyncio.gather(*coroutines)
            elif RUN_TYPE.upper() == "E2E_TEST":
                coroutines = [run_e2e_test(device, TEST_RESULT_LIST, ) for device in DEVICE_LIST]
                # Gather and execute the coroutines
                await asyncio.gather(*coroutines)
            # Print the Test Results
            print("Test Results:", TEST_RESULT_LIST)
            result = all(test_status['status'].upper() == 'SUCCESS' for test_status in TEST_RESULT_LIST)
            print(result)
            if result:
                print("pipeline_run_status=True")
                sys.exit(0)
            print("pipeline_run_status=False")
            sys.exit(0)
        print("No devices were selected, check the device tag.")
        sys.exit(0)
    except Exception:
        print("pipeline_run_status=False")
        sys.exit(0)

# Run the main function
if __name__ == '__main__':
    DEVICE_LIST = sys.argv[1].strip().replace('\r', '').split(' ')
    print(DEVICE_LIST)
    RUN_TYPE = sys.argv[2]
    IOTHUB_NAME = sys.argv[3]
    TEST_DOCKER_NAME = sys.argv[4]
    PORT_NUMBER = sys.argv[5]
    MODEL_TYPE = sys.argv[6]
    DURATION = sys.argv[7]
    INTERVAL = sys.argv[8]
    asyncio.run(main())
