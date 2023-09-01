import multiprocessing
import queue
from typing import Any

from src.common import constants


class ProcessQueue:
    """
    Class for a queue for communication between processes.
    """

    def __init__(self) -> None:
        """
        Initialize the process queue with a max size of 1.
        """
        self.max_size = constants.QUEUE_MAX_SIZE
        self.shared_queue = multiprocessing.Queue(self.max_size)
        self.process_lock = multiprocessing.Lock()
        self.lock_timeout = constants.LOCK_TIMEOUT
        self.get_timeout = constants.QUEUE_GET_TIMEOUT
        self.put_timeout = constants.QUEUE_PUT_TIMEOUT

    def add_item(self, item: Any) -> None:
        """
        Add item to the queue.

        @param:
            item (Any): object to be added to the queue
        """
        if self.process_lock.acquire(timeout=self.lock_timeout):
            self.clear_item()
            try:
                self.shared_queue.put(item, block=True, timeout=self.put_timeout)
            except queue.Full:
                self.process_lock.release()
                pass
            self.process_lock.release()

    def get_item(self) -> Any:
        """
        Get item from the queue.

        @return:
            item (Any): object retrieved from the queue
        """
        try:
            item = self.shared_queue.get(block=True, timeout=self.get_timeout)
            return item
        except queue.Empty:
            return None

    def clear_item(self) -> None:
        """
        Clear an item if the queue is full.
        """
        while self.shared_queue.full():
            try:
                self.shared_queue.get_nowait()
            except queue.Empty:
                pass
