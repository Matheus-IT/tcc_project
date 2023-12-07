import time
import functools
from pyinstrument import Profiler


BASE_DIR = '.'


class TimeElapseMeasure:
    """Utility to be used as a context manager to calculate time execution"""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.finish = time.perf_counter()
        print(f'Time elapse {self.finish - self.start:0.4f} s')


def time_elapse_measure(func):
    """Utility to calculate time execution as a decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        finish = time.perf_counter()

        print(f'Time elapse: {finish - start:0.4f} s')
        return value
    return wrapper


class SampleTimeExpensiveCalls:
    """
    Utility to sample most time consuming calls as a context manager
    ATTENTION: Doesn't work well with multiple threads
    """

    def __init__(self, open_in_browser=False):
        self.output_html = open_in_browser

    def __enter__(self):
        self.profiler = Profiler()
        self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.profiler.stop()
        if self.output_html:
            output = self.profiler.output_html()
            with open(f"{BASE_DIR}/profiler_output.html", "w") as f:
                f.write(output)
        else:
            self.profiler.print(color=True)


def sample_time_expensive_calls(output_html=False):
    """
    Utility to sample most time consuming calls as a decorator
    ATTENTION: Doesn't work well with multiple threads
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler()
            profiler.start()

            result = func(*args, **kwargs)

            profiler.stop()

            if not output_html:
                profiler.print(color=True)
                return result

            output = profiler.output_html()
            with open(f"{BASE_DIR}/profiler_output.html", "w") as f:
                f.write(output)
            return result
        return wrapper
    return decorator
