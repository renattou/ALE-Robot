import sys
import os
import cv2
import numpy as np
from robot import Robot
from fpscontrol import FPSControl

class Environment(object):
    def __init__(self, args):
        self.mode = "test"
        self.screen_width = args.screen_width
        self.screen_height = args.screen_height
        self.display_screen = args.display_screen
        self.display_processed_screen = args.display_processed_screen
        self.fps_control = FPSControl(args)

    def action_dim(self):
        # Return dimension of action space
        raise NotImplementedError

    def state_dim(self):
        # Return dimension of state space
        return (self.screen_width, self.screen_height)

    def reset(self):
        # Reset environment
        raise NotImplementedError

    def step(self, action):
        # Perform action and returns reward
        raise NotImplementedError

    def set_mode(self, mode):
        # Set training/test mode. Not used in Gym environment
        self.mode = mode

    def _get_state(self, screen):
        screen = self._correct_color(screen)
        state = self._preprocess_screen(screen)
        state = np.array(state, np.uint8)
        assert state.shape == self.state_dim()
        return state

    def _preprocess_screen(self, screen):
        # Display screen if desired
        if self.display_screen:
            self._render_screen(screen)
        # Convert from RGB to gray
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        # Resize image to correct dimensions
        screen = cv2.resize(screen, (self.screen_width, self.screen_height), interpolation=cv2.INTER_LINEAR)
        # Display processed screen if desired
        if self.display_processed_screen:
            self._render_processed_screen(screen)
        return screen

    def _correct_color(self, screen):
        return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    def _render_screen(self, screen):
        # Render screen image
        cv2.imshow('environment1', screen)
        cv2.moveWindow('environment1', 10, 20)
        cv2.waitKey(1)

    def _render_processed_screen(self, screen):
        # Render screen image
        cv2.imshow('environment2', screen)
        cv2.moveWindow('environment2', 180, 20)
        cv2.waitKey(1)

class ALEEnvironment(Environment):
    def __init__(self, rom_file, args):
        from ale_python_interface import ALEInterface
        self.ale = ALEInterface()

        # Set ALE configuration
        self.ale.setInt(b'frame_skip', args.frame_skip)
        self.ale.setFloat(b'repeat_action_probability', args.repeat_action_probability)
        self.ale.setBool(b'color_averaging', args.color_averaging)

        if args.random_seed:
            self.ale.setInt(b'random_seed', args.random_seed)

        if args.record_screen_path:
            if not os.path.exists(args.record_screen_path):
                os.makedirs(args.record_screen_path)
            self.ale.setString(b'record_screen_dir', args.record_screen_path.encode())

        if args.record_sound_filename:
            self.ale.setBool(b'sound', True)
            self.ale.setString(b'record_sound_filename', args.record_sound_filename.encode())

        # Load ROM
        self.ale.loadROM(rom_file.encode())

        # Set game difficulty and mode (after loading)
        self.ale.setDifficulty(args.game_difficulty)
        self.ale.setMode(args.game_mode)

        # Whether to use minimum set or set
        if args.minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
        else:
            self.actions = self.ale.getLegalActionSet()

        # Life lost control
        self.life_lost = False

        # Initialize base class
        super(ALEEnvironment, self).__init__(args)

    def action_dim(self):
        return len(self.actions)

    def reset(self):
        # In test mode, the game is simply initialized. In train mode, if the game
        # is in terminal state due to a life loss but not yet game over, then only
        # life loss flag is reset so that the next game starts from the current
        # state. Otherwise, the game is simply initialized.
        if (
                self.mode == 'test' or
                not self.life_lost or # `reset` called in a middle of episode
                self.ale.game_over() # all lives are lost
        ):
            self.ale.reset_game()
        self.life_lost = False
        screen = self._get_state(self.ale.getScreenRGB())
        return screen

    def step(self, action, action_b=0, ignore_screen=False):
        lives = self.ale.lives()
        # Act on environment
        reward = self.ale.act(self.actions[action], self.actions[action_b]+18)
        # Check if life was lost
        self.life_lost = (not lives == self.ale.lives())
        # Check terminal state
        terminal = (self.ale.game_over() or self.life_lost) if self.mode == 'train' else self.ale.game_over()
        # Check if should ignore the screen (in case of RobotEnvironment)
        if ignore_screen:
            screen = None
        else:
            # Get screen from ALE
            screen = self._get_state(self.ale.getScreenRGB())
            # Wait for next frame to start
            self.fps_control.wait_next_frame()
        return screen, reward, terminal

class GymEnvironment(Environment):
    def __init__(self, env_id, args):
        import gym
        self.gym = gym.make(env_id)

        # Initialize base class
        super(GymEnvironment, self).__init__(args)

    def action_dim(self):
        import gym
        assert isinstance(self.gym.action_space, gym.spaces.Discrete)
        return self.gym.action_space.n

    def reset(self):
        screen = self.gym.reset()
        screen = self._correct_color(screen)
        screen = self._get_state(screen)
        return screen

    def step(self, action):
        # Act on environment
        screen, reward, terminal, _ = self.gym.step(action)
        screen = self._correct_color(screen)
        screen = self._get_state(screen)
        # Wait for next frame to start
        self.fps_control.wait_next_frame()
        return screen, reward, terminal

class RobotEnvironment(ALEEnvironment):
    def __init__(self, rom_file, args):
        # Initialize ALE environment
        super(RobotEnvironment, self).__init__(rom_file, args)

        # Read arguments
        self.wait_action = args.wait_action
        self.enable_sync = args.enable_sync

        # Initialize Robot
        resolution = self.ale.getScreenDims()
        self.robot = Robot(self.actions, resolution, args)

        # Time controlled by simulation if synchronous
        if self.enable_sync:
            self.robot.wait_next_step()
            self.fps_control.time_function = self.robot.sim.get_last_cmd_time
            self.fps_control.start_frame()

    def reset(self):
        super(RobotEnvironment, self).reset()
        self.robot.restart()

        # Do one step so buffers have data
        if self.enable_sync:
            self.robot.wait_next_step()
        # self.robot.wait_buffers()

        # Start first frame
        self.fps_control.start_frame()

        # Get initial screen
        screen = self.robot.get_image_from_vrep()
        screen = self._get_state(screen)
        return screen

    def step(self, action, action_b=0):
        # Execute network action on simulator
        self.robot.set_joints_position_by_action(action)

        # Send action to robot and maybe wait for robot to execute action
        reward = 0
        robot_action = None
        while action != robot_action:
            # Get action being performed by the robot
            robot_action = self.robot.get_action_value()
            # print(robot_action, "->", action)

            # Execute robot action on emulator
            _, partial_reward, terminal = super(RobotEnvironment, self).step(robot_action, action_b, ignore_screen=True)
            reward += partial_reward

            # Send ALE screen to simulator
            self.robot.send_image_to_vrep(self.ale.getScreenRGB())

            # Sync with V-REP if sync is enabled
            if self.enable_sync:
                # Wait until step over so we get an updated image from the camera
                self.robot.wait_next_step()
                # print(self.robot.sim.get_last_cmd_time())

            # Get screen from robot camera
            screen = self.robot.get_image_from_vrep()
            screen = self._get_state(screen)

            # Wait for next frame to start
            self.fps_control.wait_next_frame()

            # Stop in case of terminal state or in case we don't want to wait the action to be executed
            if terminal or not self.wait_action:
                break
        return screen, reward, terminal
