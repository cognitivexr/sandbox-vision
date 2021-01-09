import time

class Timer():
    """Utility function to track the execution times"""
    def __init__(self):
        self._start_time = None
        self._count = 0
        self._average = 0
        self._longest = -float('inf')
        self._shortest = float('inf')
        self._elapsed_time = 0
        self.fps = 0

    def start(self):
        """Start the timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._elapsed_time = elapsed_time
        self._start_time = None
        self._count = self._count+1
        if elapsed_time>self._longest:
            self._longest=elapsed_time
        if elapsed_time<self._shortest:
            self._shortest=elapsed_time
        self._average=((self._count-1)/self._count)*self._average+(1/self._count)*elapsed_time
        self.fps = 1/self._average
    
    def print_summary(self):
        self.fps = 1/self._average
        print(f'time: {self._elapsed_time} fps: {self.fps} average: {self._average} shortest: {self._shortest} longest: {self._longest} count: {self._count}')

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

