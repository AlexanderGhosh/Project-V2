import time


class Timer:
    timerCount = 1

    def __init__(self, name):
        self.start_ = 0
        self.name = name or f'Timer {Timer.timerCount}'
        self.timerCount += 1

    def start(self):
        self.start_ = time.process_time()
        # print(f'{self.name} started')

    def stop(self):
        duration = time.process_time() - self.start_
        print(f'{self.name}: {duration} secconds')

    def __enter__(self):
        self.start()
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
