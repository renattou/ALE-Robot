import os, random
from collections import deque
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # Use only needed GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Use only 50% of GPU memory
tf.Session(config=config)
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import losses
import utils
from memory import ReplayMemory, StateBuffer

class DQN(object):
    def __init__(self, state_dim, action_dim, args):
        # Initialize variables from parameters and args
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.input_dim = (args.history_length,) + self.state_dim
        self.batch_size = args.batch_size
        self.gamma = args.discount_rate
        self.enable_double = args.enable_double

        # Create model and target model
        self.optimizer = self._get_optimizer(args)
        self.loss = self._get_loss(args)
        self.model = self._create_model()
        self.target_model = self._create_model()

        # Initialize memory
        self.memory = ReplayMemory(args)
        self.buffer = StateBuffer(args)

        # Statistics
        self.callback = None
        self.train_iterations = 0

    def _get_optimizer(self, args):
        if args.optimizer == 'rmsprop':
            return RMSprop(lr=args.learning_rate, rho=args.optimizer_decay, epsilon=args.optimizer_epsilon)
        elif args.optimizer == 'adam':
            return Adam(lr=args.learning_rate, epsilon=args.optimizer_epsilon)
        elif args.optimizer == 'adadelta':
            return Adadelta(lr=args.learning_rate, rho=args.optimizer_decay, epsilon=args.optimizer_epsilon)
        else:
            assert False, "Unknown optimizer"

    def _get_loss(self, args):
        if args.loss == 'mse':
            return losses.mean_squared_error
        elif args.loss == 'huber':
            return utils.huber_loss
        else:
            assert False, "Unknown loss function"

    def _create_model(self):
        # Build model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=self.input_dim, data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.action_dim, activation='linear'))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def load(self, filepath):
        # Load weights from filepath
        if os.path.isfile(filepath):
            self.model.load_weights(filepath)
            self.update_target_model()
        else:
            print("File to load doesn't exist. Doing nothing.")

    def save(self, filepath):
        # Save weights from filepath
        self.model.save_weights(filepath)

    def remember(self, action, reward, screen, terminal, training=True):
        # Store current screen to screen history buffer
        self.buffer.add(screen)
        # Store current (action, reward, screen, terminal) to memory if training only
        if training:
            self.memory.add(action, reward, screen, terminal)

    def forward(self, epsilon):
        # Choose next action
        if np.random.rand() <= epsilon:
            # Randomize action
            return random.randrange(self.action_dim)
        else:
            # Predict action
            state = self.buffer.get_state_minibatch()
            q_values = self.model.predict_on_batch(state)
            assert len(q_values[0]) == self.action_dim
            return np.argmax(q_values[0])

    def backward(self):
        # Get mini-batch
        minibatch = self.memory.get_minibatch()
        prestate_batch, action_batch, reward_batch, poststate_batch, terminal_batch = minibatch
        assert len(prestate_batch.shape) == 4
        assert len(poststate_batch.shape) == 4
        assert len(action_batch.shape) == 1
        assert len(reward_batch.shape) == 1
        assert len(terminal_batch.shape) == 1
        assert prestate_batch.shape == poststate_batch.shape
        assert prestate_batch.shape[0] == action_batch.shape[0] == reward_batch.shape[0] == poststate_batch.shape[0] == terminal_batch.shape[0]

        # Compute Q values for mini-batch update
        if self.enable_double:
            # Predict actions from online network and extract the maximum for each sample
            q_values = self.model.predict_on_batch(poststate_batch)
            assert q_values.shape == (self.batch_size, self.action_dim)
            actions = np.argmax(q_values, axis=1)

            # Predict Q values from target network and extract highest values in relation to "actions"
            target_q_values = self.target_model.predict_on_batch(poststate_batch)
            assert target_q_values.shape == (self.batch_size, self.action_dim)
            q_batch = target_q_values[range(self.batch_size), actions]
        else:
            # Predict Q values from target network and extract the maximum for each sample
            target_q_values = self.target_model.predict_on_batch(poststate_batch)
            assert target_q_values.shape == (self.batch_size, self.action_dim)
            q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (self.batch_size,)

        # Compute target values for each sample (only for taken actions)
        target_batch = self.model.predict_on_batch(prestate_batch)
        for i in range(self.batch_size):
            if terminal_batch[i]:
                # target = reward  (for terminal state_[t+1])
                target_batch[i, action_batch[i]] = reward_batch[i]
            else:
                # target = reward + gamma * Q(state_[t+1])  (for non-terminal state_[t+1])
                target_batch[i, action_batch[i]] = reward_batch[i] + self.gamma * q_batch[i]
        assert target_batch.shape == (self.batch_size, self.action_dim)

        # Train online network
        loss = self.model.train_on_batch(prestate_batch, target_batch)

        # Calculate statistics
        self.train_iterations += 1
        if self.callback:
            self.callback.on_train(loss)
