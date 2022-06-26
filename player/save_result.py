from queue import Queue

class save_result:
    def __init__(self):
        self._time = 0
        self._memory_max_time = int(time / 0.005)
        self._sensing = 0
        self._success = 0
        self._failure = 0
        self._reward = 0

    def init_result(self):
        self._time = 0
        self._sensing = 0
        self._success = 0
        self._failure = 0
        self._reward = 0

    def get_result(self, time, success, failure, reward):
        self._time = 0