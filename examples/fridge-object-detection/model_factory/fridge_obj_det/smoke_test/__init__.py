"""A module defining the Fridge Object Detection smoke test package."""

from model_factory.fridge_obj_det.smoke_test.fridge_test_module import FridgeTestModule


def get_instance(port):
    """Get an instance of the FridgeTestModule class.

    Args:
        port (int): The port number to use for the smoke test.

    Returns:
        FridgeTestModule: An instance of the FridgeTestModule class.
    """
    return FridgeTestModule(port)
