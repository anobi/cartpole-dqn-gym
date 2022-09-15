import random
from collections import namedtuple

Transition = namedtuple(
    'Transition', [
        'state',
        'action',
        'next_state',
        'reward'
    ]
)

#class Transition():
    #def __init__():
        #pass

class Memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
