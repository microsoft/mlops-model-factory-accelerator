from multiprocessing import Process
from multiprocessing.context import DefaultContext
import logging


from src.common.config_handler import ConfigHandler
from src.frame_processor.inference_result_handler import DefaultInferenceResultHandler
from src.multiprocessing.frame_capture_process import FrameCaptureProcess
from src.multiprocessing.edge_inferencing_process import EdgeInferencingProcess
from src.multiprocessing.inference_result_handler_process import InferenceResultHandlerProcess
from src.multiprocessing.queue import ProcessQueue


class ProcessController:
    """
    Class for starting, stopping and restarting processes
    """

    def __init__(self, multiprocessing_context: DefaultContext) -> None:
        """
        Initialize process controller.

        @param:
            multiprocessing_context (DefaultContext): Multiprocessing context
        """
        self.logger = logging.getLogger(__name__)
        self.multiprocessing_context = multiprocessing_context

    def start(self, **kwargs) -> bool:
        """
        Start the processes.

        @return:
            success (bool): True if the processes are started successfully, False otherwise
        """
        override_video_source = kwargs.get('video_source')
        override_inference_result_handler = kwargs.get('inference_result_handler')

        self.frame_capture_process = None
        self.edge_inferencing_process = None
        self.inference_result_handler_process = None
        self.inference_result_handler = None

        self.config_handler = ConfigHandler()
        self.frame_queue = ProcessQueue()
        self.inference_queue = ProcessQueue()

        self.logger.info("Starting all processes...")
        if (
            self.frame_capture_process is not None
            and self.frame_capture_process.is_alive()
        ):
            self.logger.info("Frame capture process is already running...")
            return False
        if (
            self.edge_inferencing_process is not None
            and self.edge_inferencing_process.is_alive()
        ):
            self.logger.info("Edge inferencing process is already running...")
            return False
        if (
            self.inference_result_handler_process is not None
            and self.inference_result_handler_process.is_alive()
        ):
            self.logger.info("Inference result handler process is already running...")
            return False

        # Start frame capture process
        frame_capture_config = self.config_handler.get_frame_capture_config(override_video_source)
        self.frame_capture_process = self.multiprocessing_context.Process(
            target=FrameCaptureProcess().run,
            args=(frame_capture_config, self.frame_queue,),
            name="frame_capture_process",
        )
        self.frame_capture_process.daemon = True
        self.frame_capture_process.start()

        # Start edge inferencing process
        edge_inference_config = self.config_handler.get_edge_inference_config()
        self.edge_inferencing_process = self.multiprocessing_context.Process(
            target=EdgeInferencingProcess().run,
            args=(edge_inference_config, self.frame_queue, self.inference_queue,),
            name="edge_inferencing_process",
        )
        self.edge_inferencing_process.daemon = True
        self.edge_inferencing_process.start()

        # Start inference result handler process
        self.set_inference_result_handler(override_inference_result_handler)
        self.inference_result_handler_process = self.multiprocessing_context.Process(
            target=InferenceResultHandlerProcess().run,
            args=(self.inference_result_handler,),
            name="inference_result_handler_process"
        )
        self.inference_result_handler_process.daemon = True
        self.inference_result_handler_process.start()
        return True

    def stop(self) -> bool:
        """
        Stop the processes.

        @return:
            success (bool): True if the processes are stopped successfully, False otherwise
        """
        self.logger.info("Stopping all processes...")
        if (
            self.frame_capture_process is not None
            and self.frame_capture_process.is_alive()
        ):
            self._terminate_process(self.frame_capture_process)
            self.frame_capture_process = None
            self.logger.info("Frame capture process has been stopped...")
        if (
            self.edge_inferencing_process is not None
            and self.edge_inferencing_process.is_alive()
        ):
            self._terminate_process(self.edge_inferencing_process)
            self.edge_inferencing_process = None
            self.logger.info("Edge inferencing process has been stopped...")

        if (
            self.inference_result_handler_process is not None
            and self.inference_result_handler_process.is_alive()
        ):
            self._terminate_process(self.inference_result_handler_process)
            self.inference_result_handler_process = None
            self.logger.info("Inference result handler process has been stopped...")

        self.frame_queue.shared_queue.close()
        self.inference_queue.shared_queue.close()
        self.logger.info("Stopped all processes...")
        return True

    def restart(self) -> bool:
        """
        Restart the processes.

        @return:
            success (bool): True if the processes are restarted successfully, False otherwise
        """
        self.logger.info("Restarting all processes...")
        self.stop()
        result = self.start()
        self.logger.info("Restarted all processes...")
        return result

    def _terminate_process(self, process: Process, retry_count: int = 0) -> None:
        """
        Terminate the process with retry (internal).

        @param:
            process (Process): Process object
            retry_count (int): Retry count
        """
        self.logger.info(f"Terminating process {process.name}...")
        while retry_count < 3:
            try:
                process.terminate()
                return  # Process terminated successfully, exit the loop
            except Exception as e:
                self.logger.exception(e)
                retry_count += 1
                self.logger.info(f"Retrying process termination... Retry count: {retry_count}")
        self.logger.error(f"Could not terminate process {process.name}...")
        raise Exception(f"Could not terminate process {process.name}...")

    def set_inference_result_handler(self, override_inference_result_handler=None):
        if override_inference_result_handler is None:
            self.inference_result_handler = DefaultInferenceResultHandler(self.inference_queue)
        else:
            self.inference_result_handler = override_inference_result_handler(self.inference_queue)
