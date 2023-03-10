import time


class Timer:
    timerCount = 1

    def __init__(self, name):
        self.start_ = 0
        self.name = name or f'Timer {Timer.timerCount}'
        self.timerCount += 1

    def start(self):
        self.start_ = time.process_time_ns()
        print(f'{self.name} started')

    def end(self):
        duration = time.process_time_ns() - self.start_
        print(f'{self.name}: {duration * 1e-6} ms')

    def __enter__(self):
        self.start()
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
