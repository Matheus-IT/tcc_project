from timeit import default_timer as timer
import functools
from pyinstrument import Profiler


BASE_DIR = '.'


class Timer:
    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *exc_info):
        end = timer()
        print(end - self.start)


class SampleTimeExpensiveCalls:
    """
    Utility to sample most time consuming calls as a context manager
    ATTENTION: Doesn't work well with multiple threads
    """

    def __init__(self, open_in_browser=False):
        self.open_in_browser = open_in_browser

    def __enter__(self):
        self.profiler = Profiler()
        self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.open_in_browser:
            self.profiler.open_in_browser()
            return
        self.profiler.print()


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
