import random
import numpy as np

class Agent(object):
    def __init__(self, env, net, args):
        self.env = env
        self.net = net

        # Training epsilon
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.exploration_rate_test = args.exploration_rate_test
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_decay = (args.exploration_rate_start - args.exploration_rate_end) / args.exploration_decay_steps
        self.total_train_steps = 0

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        self.random_starts = args.random_starts
        self.history_length = args.history_length

        # Statistics
        self.callback = None

    def _epsilon(self):
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * self.exploration_decay
        else:
            return self.exploration_rate_end

    def _reset_random(self):
        # Reset environment
        self.env.reset()

        # Perform random number of dummy actions to produce more stochastic games
        for _ in range(random.randint(self.history_length, self.random_starts) + 1):
            noop = 0
            screen, reward, terminal = self.env.step(noop)
            if terminal:
                self.env.restart()
            # Add dummy states to buffer
            self.net.remember(noop, reward, screen, terminal, training=False)

    def _step(self, exploration_rate, action_b=0, training=True):
        # Predict action
        action = self.net.forward(exploration_rate)

        # Execute action on environment
        screen, reward, terminal = self.env.step(action, action_b)

        # Save current data to memory and buffer
        self.net.remember(action, reward, screen, terminal, training=training)

        # Reset after terminal state
        if terminal:
            # Reset environment
            self._reset_random()

        # Calculate statistics
        if self.callback:
            self.callback.on_step(action, reward, terminal, screen, exploration_rate)

        return action, reward, terminal, screen

    def play_random(self, random_steps):
        # Reset environment
        self.env.reset()

        # Play given number of steps
        for _ in range(random_steps):
            # Do random action
            _, _, _, _ = self._step(1)

    def train(self, train_steps):
        # Reset environment
        self._reset_random()
        total_reward = 0.0

        # Train for given number of steps
        for i in range(train_steps):
            # Count current step
            self.total_train_steps += 1

            # Execute step on agent
            _, reward, terminal, _ = self._step(self._epsilon())
            total_reward += reward

            # Update target model
            if self.target_steps >= 1 and i % self.target_steps == 0:
                self.net.update_target_model()

            # Train network
            if i % self.train_frequency == 0:
                for _ in range(self.train_repeat):
                    self.net.backward()

            # Print results on terminal state
            if terminal:
                print("Train episode ended with score: " + str(total_reward) + " on step " + str(self.total_train_steps))
                total_reward = 0.0

    def test(self, test_steps):
        # Reset environment
        self._reset_random()
        total_reward = 0.0

        # Test for given number of steps
        for _ in range(test_steps):
            # Execute step on agent
            _, reward, terminal, _ = self._step(self.exploration_rate_test, training=False)
            total_reward += reward

            # Print results on terminal state
            if terminal:
                print("Test episode ended with score: " + str(total_reward))
                total_reward = 0.0

    def play(self):
        # Reset environment
        self._reset_random()
        terminal = False
        total_reward = 0.0

        while not terminal:
            # Execute step on agent
            _, reward, terminal, _ = self._step(self.exploration_rate_test, training=False)
            total_reward += reward

        # Print results
        print("Play episode ended with score: " + str(total_reward))

    def play_two_players(self, player_b):
        # Reset environment
        self._reset_random()
        terminal = False
        total_reward = 0.0

        while not terminal:
            # Get action from player
            action_b = player_b.get_action()
            # End play if user wants to exit
            if action_b == -1:
                break

            # Execute step on agent
            _, reward, terminal, _ = self._step(self.exploration_rate_test, action_b=action_b, training=False)
            total_reward += reward

            # Get image from emulator and render it to user
            screen = self.env.ale.getScreenRGB()
            player_b.render_screen(screen)

        # Print results
        print("2-player episode ended with score: " + str(total_reward))
