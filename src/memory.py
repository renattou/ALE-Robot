import numpy as np
import random

# From: https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py
class ReplayMemory(object):
  def __init__(self, args):
    self.size = args.replay_size
    # preallocate memory
    self.actions = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size, dtype = np.integer)
    self.screens = np.empty((self.size, args.screen_height, args.screen_width), dtype = np.uint8)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
  
  def get_state(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def get_minibatch(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.get_state(index - 1)
      self.poststates[len(indexes), ...] = self.get_state(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals

# From: https://github.com/tambetm/simple_dqn/blob/master/src/state_buffer.py
class StateBuffer(object):
  """
  While ReplayMemory could have been used for fetching the current state,
  this also means that test time states make their way to training process.
  Having separate StateBuffer ensures that test data doesn't leak into training.
  """
  def __init__(self, args):
    self.history_length = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.buffer = np.zeros((self.batch_size, self.history_length) + self.dims, dtype=np.uint8)

  def add(self, screen):
    assert screen.shape == self.dims
    self.buffer[0, :-1] = self.buffer[0, 1:]
    self.buffer[0, -1] = screen

  def get_state(self):
    return self.buffer[0]

  def get_state_minibatch(self):
    return self.buffer

  def reset(self):
    self.buffer *= 0
