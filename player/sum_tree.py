import numpy as np


class sumTree:
    write = 0

    def __init__(self, memory_size):
        self._memory_size = memory_size
        self._tree = np.zeros(2*memory_size - 1)
        self._data = np.zeros(memory_size, dtype=object)
        self._entry = 0

    @property
    def entry(self):
        return self._entry

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self._tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

# find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def total(self):
        return self._tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self._memory_size - 1

        self._data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self._memory_size:
            self.write = 0

        if self._entry < self._memory_size:
            self._entry += 1

    # update priority
    def update(self, idx, p):
        change = p - self._tree[idx]

        self._tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self._memory_size + 1

        return idx, self._tree[idx], self._data[dataIdx]