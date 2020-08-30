import contextlib
import logging
from datetime import datetime

# class Timer:
#     def __init__(self, label: str, logger: logging.Logger):
#         self.label = label
#         self.logger = logger

#     def __enter__(self) -> None:
#         self.start_time = datetime.now()

#     def __exit__(self, exc_type, exc_val, exc_tb) -> None:
#         self.logger.debug("Time for %s: %s", self.label, datetime.now() - self.start_time)


@contextlib.contextmanager
def timer(label: str, logger: logging.Logger) -> None:
    start_time = datetime.now()
    yield
    logger.info("Time for %s: %s", label, datetime.now() - start_time)
