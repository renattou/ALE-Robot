import math
import cv2
import numpy as np
from simulator import Simulator

VISION_SENSOR = 'projectorVisionSensor'
CAMERA_SENSOR = 'NAO_vision1'
FIRE_SIGNAL = 'JoystickFire'
LEFT_SIGNAL = 'JoystickLeft'
RIGHT_SIGNAL = 'JoystickRight'
UP_SIGNAL = 'JoystickUp'
DOWN_SIGNAL = 'JoystickDown'
R_SHOULDER_PITCH = 'RShoulderPitch3'
L_SHOULDER_PITCH = 'LShoulderPitch3'
L_SHOULDER_ROLL = 'LShoulderRoll3'
L_ELBOW_ROLL = 'LElbowRoll3'

class Robot(object):
    def __init__(self, actions, resolution, args):
        # Start simulator
        self.sim = Simulator(args)

        # Process arguments
        self.actions = actions
        self.display_camera = args.display_camera
        self.display_scale = args.camera_screen_scale
        self.size = width, height = resolution
        self.rgb_size = self.size + (3,)
        self.pts_dst = np.array([[0, 0], [0, height], [width, height], [width, 0]], np.uint8)
        self.step_time = 66.7 / 1000

        # Get vision and camera sensor handles
        self.vision_sensor = self.sim.get_handle(VISION_SENSOR)
        self.camera_sensor = self.sim.get_handle(CAMERA_SENSOR)

        # Get joint handles
        self.r_shoulder_pitch = self.sim.get_handle(R_SHOULDER_PITCH)
        self.l_shoulder_pitch = self.sim.get_handle(L_SHOULDER_PITCH)
        self.l_shoulder_roll = self.sim.get_handle(L_SHOULDER_ROLL)
        self.l_elbow_roll = self.sim.get_handle(L_ELBOW_ROLL)

        # Initialize data streams
        self.sim.get_vision_sensor_image(self.camera_sensor, first_call=True)
        self.sim.get_integer_signal(FIRE_SIGNAL, first_call=True)
        self.sim.get_integer_signal(LEFT_SIGNAL, first_call=True)
        self.sim.get_integer_signal(RIGHT_SIGNAL, first_call=True)
        self.sim.get_integer_signal(UP_SIGNAL, first_call=True)
        self.sim.get_integer_signal(DOWN_SIGNAL, first_call=True)
        self.sim.get_joint_position(self.r_shoulder_pitch, first_call=True)
        self.sim.get_joint_position(self.l_shoulder_pitch, first_call=True)
        self.sim.get_joint_position(self.l_shoulder_roll, first_call=True)
        self.sim.get_joint_position(self.l_elbow_roll, first_call=True)

    def restart(self):
        # Restart simulator
        self.sim.restart_simulation()

    def wait_next_step(self):
        # Trigger next step
        self.sim.step_simulation()
        # Wait until step is over to read updated data
        self.sim.get_ping_time()

    def wait_buffers(self):
        resolution, _ = self.sim.get_vision_sensor_image(self.camera_sensor)
        while not resolution:
            resolution, _ = self.sim.get_vision_sensor_image(self.camera_sensor)

    def send_image_to_vrep(self, image):
        image = cv2.copyMakeBorder(image, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        array_image = image.flatten()
        self.sim.set_vision_sensor_image(self.vision_sensor, array_image)

    def get_image_from_vrep(self):
        resolution, image = self.sim.get_vision_sensor_image(self.camera_sensor)
        image = np.array(image, dtype=np.uint8)
        image.resize([resolution[1], resolution[0], 3])
        image = cv2.flip(image, 0)
        if self.display_camera:
            self._render_camera(image)
        screen_image = self._extract_screen(image)
        return screen_image

    def _render_camera(self, image):
        image = cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('camera', image)
        cv2.moveWindow('camera', 275, 20)
        cv2.waitKey(1)

    def _extract_screen(self, image):
        result = image

        # Try to mask out the screen
        projector_rect = self._get_biggest_rectangle(image, 10, cv2.THRESH_BINARY_INV)
        if projector_rect is None:
            # print("Screen not detected by robot.")
            return np.zeros(self.rgb_size, np.uint8)
        else:
            mask = np.zeros_like(result)
            cv2.drawContours(mask, [projector_rect], -1, (255, 255, 255), cv2.FILLED)
            result = cv2.bitwise_and(result, mask)

        # Try to mask out the screen
        screen_rect = self._get_biggest_rectangle(result, 30, cv2.THRESH_BINARY)
        if screen_rect is None:
            # print("Screen not detected by robot.")
            return np.zeros(self.rgb_size, np.uint8)
        else:
            pts_src = screen_rect
            homography, _ = cv2.findHomography(pts_src, self.pts_dst)
            result = cv2.warpPerspective(result, homography, self.size, flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

        # Get screen countour
        return result

    def _get_biggest_rectangle(self, image, threshold, threshold_type):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, threshold_type)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        rectangle = None
        for i, cont in enumerate(contours):
            arc_len = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.01 * arc_len, True)
            if len(approx) == 4:
                area = cv2.contourArea(cont)
                if area > largest_area:
                    largest_area = area
                    rectangle = approx
        return rectangle

    def get_action_value(self):
        # Get button signals
        fire = True if self.sim.get_integer_signal(FIRE_SIGNAL) == 1 else False
        left = True if self.sim.get_integer_signal(LEFT_SIGNAL) == 1 else False
        right = True if self.sim.get_integer_signal(RIGHT_SIGNAL) == 1 else False
        up = True if self.sim.get_integer_signal(UP_SIGNAL) == 1 else False
        down = True if self.sim.get_integer_signal(DOWN_SIGNAL) == 1 else False

        # Get action from signals
        action = 0 # noop
        if up:
            if right:
                action = 6 # up-right
            elif left:
                action = 7 # up-left
            else:
                action = 2 # up
        elif down:
            if right:
                action = 8 # down-right
            elif left:
                action = 9 # down-left
            else:
                action = 5 # down
        elif right:
            action = 3 # right
        elif left:
            action = 4 # left
        if fire:
            if action == 0:
                action = 1 # fire
            else:
                action += 8 # direction + fire

        # Convert action to action index
        action_index = np.where(self.actions == action)[0]
        if len(action_index) == 0:
            # If invalid send noop action
            print("Robot is doing invalid action.")
            action_index = 0
        else:
            action_index = action_index[0]

        return action_index

    def set_joints_position_by_action(self, action_index):
        # Get target positions
        r_shoulder_pitch_pos, l_shoulder_pitch_pos, l_shoulder_roll_pos, l_elbow_roll_pos = self.get_joints_by_action(action_index)

        # Set joint target positions at once
        self.sim.pause()
        self.sim.set_joint_target_position(self.r_shoulder_pitch, r_shoulder_pitch_pos)
        self.sim.set_joint_target_position(self.l_shoulder_pitch, l_shoulder_pitch_pos)
        self.sim.set_joint_target_position(self.l_shoulder_roll, l_shoulder_roll_pos)
        self.sim.set_joint_target_position(self.l_elbow_roll, l_elbow_roll_pos)
        self.sim.resume()

    def set_joints_velocity_by_action(self, action_index):
        # Get current action
        cur_action_index = self.get_action_value()

        # If action is different, get target velocity
        if cur_action_index != action_index:
            r_shoulder_pitch_pos, l_shoulder_pitch_pos, l_shoulder_roll_pos, l_elbow_roll_pos = self.get_joints_by_action(action_index)

            self.sim.pause()
            cur_r_shoulder_pitch_pos = self.sim.get_joint_position(self.r_shoulder_pitch)
            cur_l_shoulder_pitch_pos = self.sim.get_joint_position(self.l_shoulder_pitch)
            cur_l_shoulder_roll_pos = self.sim.get_joint_position(self.l_shoulder_roll)
            cur_l_elbow_roll_pos = self.sim.get_joint_position(self.l_elbow_roll)
            self.sim.resume()

            r_shoulder_pitch_vel = (r_shoulder_pitch_pos - cur_r_shoulder_pitch_pos) / self.step_time
            l_shoulder_pitch_vel = (l_shoulder_pitch_pos - cur_l_shoulder_pitch_pos) / self.step_time
            l_shoulder_roll_vel = (l_shoulder_roll_pos - cur_l_shoulder_roll_pos) / self.step_time
            l_elbow_roll_vel = (l_elbow_roll_pos - cur_l_elbow_roll_pos) / self.step_time
        # If action is correct set velocity to zero
        else:
            r_shoulder_pitch_vel = 0
            l_shoulder_pitch_vel = 0
            l_shoulder_roll_vel = 0
            l_elbow_roll_vel = 0

        # Set joint target velocities at once
        self.sim.pause()
        self.sim.set_joint_target_velocity(self.r_shoulder_pitch, r_shoulder_pitch_vel)
        self.sim.set_joint_target_velocity(self.l_shoulder_pitch, l_shoulder_pitch_vel)
        self.sim.set_joint_target_velocity(self.l_shoulder_roll, l_shoulder_roll_vel)
        self.sim.set_joint_target_velocity(self.l_elbow_roll, l_elbow_roll_vel)
        self.sim.resume()

    def get_joints_by_action(self, action_index):
        # Convert action index to action
        action = self.actions[action_index]

        # Initial values
        r_shoulder_pitch_pos = 20
        l_shoulder_pitch_pos = 22
        l_shoulder_roll_pos = -3.5
        l_elbow_roll_pos = -30.5

        # Get joint positios
        if action == 1 or action >= 10: # fire
            r_shoulder_pitch_pos = 22
        if action == 2 or action == 10: # up
            l_shoulder_pitch_pos = 24.5
            l_shoulder_roll_pos = -18
            l_elbow_roll_pos = -4
        elif action == 3 or action == 11: # right
            l_shoulder_pitch_pos = 24.5
            l_shoulder_roll_pos = -11
            l_elbow_roll_pos = -25
        elif action == 4 or action == 12: # left
            l_shoulder_pitch_pos = 24.25
            l_shoulder_roll_pos = 0
            l_elbow_roll_pos = -32
        elif action == 5 or action == 13: # down
            l_shoulder_pitch_pos = 24
            l_shoulder_roll_pos = 4
            l_elbow_roll_pos = -45
        elif action == 6: # up-right
            l_shoulder_pitch_pos = 25
            l_shoulder_roll_pos = -14
            l_elbow_roll_pos = -14
        elif action == 7: # up-left
            l_shoulder_pitch_pos = 25
            l_shoulder_roll_pos = -7
            l_elbow_roll_pos = -20
        elif action == 8: # down-right
            l_shoulder_pitch_pos = 25
            l_shoulder_roll_pos = -4
            l_elbow_roll_pos = -35
        elif action == 9: # down-left
            l_shoulder_pitch_pos = 25
            l_shoulder_roll_pos = 5
            l_elbow_roll_pos = -42

        # Convert to radians
        r_shoulder_pitch_pos *= math.pi / 180
        l_shoulder_pitch_pos *= math.pi / 180
        l_shoulder_roll_pos *= math.pi / 180
        l_elbow_roll_pos *= math.pi / 180

        return r_shoulder_pitch_pos, l_shoulder_pitch_pos, l_shoulder_roll_pos, l_elbow_roll_pos
