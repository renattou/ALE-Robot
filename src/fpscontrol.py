import time

class FPSControl(object):
    def __init__(self, args):
        self.target_fps = -1 if (args.target_fps == -1 or args.enable_sync) else args.target_fps / args.frame_skip
        self.target_time = 0 if self.target_fps == -1 else 1.0 / self.target_fps
        self.smoothness = 1.0
        self.time_function = time.time
        self.start_time = self.time_function()
        self.first_start_time = self.start_time
        self.fps = self.instant_fps = 0

    def start_frame(self):
        self.start_time = self.time_function()

    def wait_next_frame(self):
        time_diff = self.time_function() - self.start_time
        # print(self.time_function() - self.first_start_time)

        # Wait until target time to reach target fps
        while time_diff < self.target_time or time_diff == 0:
            time_diff = self.time_function() - self.start_time

        # Calculate fps with smoothing
        self.instant_fps = 1.0 / time_diff
        self.fps = (self.instant_fps * self.smoothness) + (self.fps * (1.0 - self.smoothness))

        # Start new frame
        self.start_frame()
        # print(self.get_fps())

        # Return FPS
        return self.get_fps()

    def check_frame_end(self):
        time_diff = self.time_function() - self.start_time

        if time_diff < self.target_time:
            return False
        else:
            return True

    def get_fps(self, smoothed=True):
        return self.fps if smoothed else self.instant_fps
