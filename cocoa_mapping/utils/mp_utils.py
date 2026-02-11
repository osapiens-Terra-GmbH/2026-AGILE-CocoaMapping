"""Multiprocessing utilities."""
from typing import Any
import logging
import multiprocessing as mp
import queue


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def feed_while_checking_for_crash(item: Any, q: mp.Queue, processes: list[mp.Process], timeout: int = 10):
    """Try to feed item while checking if any processes died every timeout seconds.
    If any process dies, kill all processes and raise an exception.

    Args:
        item: The item to feed.
        q: The queue to feed.
        processes: The list of processes to check.
        timeout: The timeout for the queue.put, in seconds
    """
    while True:
        try:
            q.put(item, timeout=timeout)
            break
        except queue.Full:
            if not are_processes_alive(processes):
                kill_processes(processes)
                raise Exception("Some process died. Killing all processes and exiting.")


def consume_while_checking_if_producer_done(q: mp.Queue,
                                            producer_done: mp.Event,
                                            done_value: Any = None,
                                            timeout: int = 10
                                            ) -> Any:
    """Consume while checking if any processes died every timeout seconds
    If processes are done, return done_value.

    Args:
        q: The queue to consume.
        producer_done: The event to signal that the producer is done.
        timeout: The timeout for the queue.get, in seconds
        done_value: The value to return if the producer is done.
    """
    while True:
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            if producer_done.is_set():
                return done_value
            continue


def are_processes_alive(processes: list[mp.Process]):
    """Monitor processes.

    Args:
        processes: The list of processes to monitor.
    """
    for p in processes:
        if not p.is_alive() and p.exitcode != 0:
            return False
    return True


def kill_processes(processes: list[mp.Process]):
    """Kill processes.

    Args:
        processes: The list of processes to kill.
    """
    for p in processes:
        p.terminate()


def ensure_all_processes_terminated(processes: list[mp.Process], timeout: int = 5):
    """Make sure that the processes were terminated, if not, terminate and log if they do not stop after timeout
    It is intended to be run at the end, in the finally block of a try-finally block.

    Args:
        processes: The list of processes to ensure termination.
        timeout: The timeout for the processes to stop, in seconds. If not stopped after timeout, log a warning.
    """
    for p in processes:
        if p.is_alive():
            p.terminate()

    # Wait for them to stop
    for p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not exit after terminate()")
