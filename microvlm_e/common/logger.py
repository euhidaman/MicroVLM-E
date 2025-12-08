"""
Logging utilities for MicroVLM-E.
"""

import logging
import os
import sys
from datetime import datetime

from microvlm_e.common.dist_utils import is_main_process


def setup_logger(output_dir: str = None, name: str = "microvlm_e"):
    """
    Set up logger for MicroVLM-E.

    Args:
        output_dir: Directory to save log files. If None, only logs to stdout.
        name: Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only main process)
    if is_main_process():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (only main process)
    if output_dir is not None and is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"log_{timestamp}.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    return logger


class MetricLogger:
    """
    A metric logger that tracks and logs training metrics.
    """

    def __init__(self, delimiter: str = "  "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Update meters with new values."""
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                if k not in self.meters:
                    self.meters[k] = SmoothedValue(window_size=20)
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        entries = []
        for name, meter in self.meters.items():
            entries.append(f"{name}: {meter}")
        return self.delimiter.join(entries)

    def log_every(self, iterable, log_freq: int, header: str = ""):
        """Log metrics at regular intervals."""
        i = 0
        for obj in iterable:
            yield obj
            i += 1
            if i % log_freq == 0:
                logging.info(f"{header} [{i}] {self}")


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.deque = []
        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        """Add a new value."""
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            old = self.deque.pop(0)
            self.total -= old
        else:
            self.count += 1
        self.total += value

    @property
    def median(self):
        """Get median value."""
        sorted_values = sorted(self.deque)
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        return sorted_values[n // 2]

    @property
    def avg(self):
        """Get average value over window."""
        if len(self.deque) == 0:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        """Get global average value."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def value(self):
        """Get latest value."""
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return f"{self.median:.4f} ({self.global_avg:.4f})"

