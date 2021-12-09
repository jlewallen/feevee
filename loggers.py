from typing import List
from queue import SimpleQueue as Queue
import logging, asyncio

class SimpleAdapter(logging.LoggerAdapter):
    pass


class LocalQueueHandler(logging.handlers.QueueHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # Removed the call to self.prepare(), handle task cancellation
        try:
            self.enqueue(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging_queue() -> None:
    """
    Move log handlers to a separate thread.

    Replace handlers on the root logger with a LocalQueueHandler,
    and start a logging.QueueListener holding the original
    handlers.

    """
    queue: Queue = Queue()
    root = logging.getLogger()
    handlers: List[logging.Handler] = []
    handler = LocalQueueHandler(queue)
    root.addHandler(handler)
    for h in root.handlers[:]:
        if h is not handler:
            root.removeHandler(h)
            handlers.append(h)
    listener = logging.handlers.QueueListener(
        queue, *handlers, respect_handler_level=True
    )
    listener.start()
    log = get_logger("feevee")
    log.info(f"logging queue ready")


def get_logger(name: str):
    return logging.getLogger(name)
