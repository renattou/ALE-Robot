import vrep.vrep as vrep

# response values
ERROR = -1
OK = 1

# remote API script
REMOTE_API_OBJ = 'RemoteAPI'
REMOTE_API_FUNC = 'resetSimulation'
SIMULATION_TIME = 'SimulationTime'

class Simulator(object):
    def __init__(self, args):
        self.id = -1
        self.ip = args.ip
        self.port = args.port
        self.enable_sync = args.enable_sync
        self.is_connected = False
        self.is_running = False

        # Connect and start simulation
        self.connect()
        self.start_simulation()

        # Initialize data streams
        self.get_simulation_time(first_call=True)

    def check_is_connected(self):
        # Check if client is connected to server
        if not self.is_connected:
            raise RuntimeError('Client is not connected to V-REP.')

    def check_is_running(self):
        # Check if simulation is running
        if not self.is_running:
            raise RuntimeError('Simulation is not running.')

    def check_return_code(self, return_code, tolerance=vrep.simx_return_ok):
        # Check if code is different than OK and from a given tolerance
        if (return_code != vrep.simx_return_ok) and (return_code != tolerance):
            raise RuntimeError('Remote API return code: (' + str(return_code) + ')')

    def connect(self):
        # Check if client is already connected to V-REP
        if self.is_connected:
            raise RuntimeError('Client is already connected to V-REP.')

        # Connect to server
        self.id = vrep.simxStart(self.ip, self.port, True, False, 2000, 5)

        # Check if connection was successful
        if self.id == -1:
            raise RuntimeError('Unable to connect to V-REP.')

        self.is_connected = True
        return self.id

    def disconnect(self):
        self.check_is_connected()

        # Disconnect from server
        vrep.simxFinish(self.id)
        self.is_connected = False

    def pause(self):
        self.check_is_connected()

        # Pause communication
        vrep.simxPauseCommunication(self.id, True)

    def resume(self):
        self.check_is_connected()

        # Resume communication
        vrep.simxPauseCommunication(self.id, False)

    def start_simulation(self):
        self.check_is_connected()

        # Check if simulation is already running
        if self.is_running:
            raise RuntimeError('Simulation is already running.')

        # Set if communication is synchronous
        if self.enable_sync:
            status = vrep.simxSynchronous(self.id, self.enable_sync)
            self.check_return_code(status)

        # Start simulation
        status = vrep.simxStartSimulation(self.id, vrep.simx_opmode_blocking)
        self.check_return_code(status)

        # Wait until simulation really started
        while True:
            vrep.simxClearIntegerSignal(self.id, "DummySignal", vrep.simx_opmode_oneshot_wait)
            _, info = vrep.simxGetInMessageInfo(self.id, vrep.simx_headeroffset_server_state)
            begin_running = info & 1
            if begin_running:
                break
        self.is_running = True

    def stop_simulation(self):
        self.check_is_running()

        # Stop simulation
        status = vrep.simxStopSimulation(self.id, vrep.simx_opmode_blocking)
        self.check_return_code(status)

        # Wait until simulation really stopped
        while True:
            vrep.simxClearIntegerSignal(self.id, "DummySignal", vrep.simx_opmode_oneshot_wait)
            _, info = vrep.simxGetInMessageInfo(self.id, vrep.simx_headeroffset_server_state)
            still_running = info & 1
            if not still_running:
                break
        self.is_running = False

    def step_simulation(self):
        self.check_is_running()

        # Trigger next step
        status = vrep.simxSynchronousTrigger(self.id)
        self.check_return_code(status)

    def restart_simulation(self):
        # Stop and start simulation again
        if self.is_running:
            self.stop_simulation()
        self.start_simulation()

    def execute_script(self, object_name, function_name):
        self.check_is_running()

        # Run script on simulation
        status, _, _, _, _ = vrep.simxCallScriptFunction(
            self.id, object_name, vrep.sim_scripttype_customizationscript,
            function_name, [], [], [], bytearray(), vrep.simx_opmode_blocking)
        self.check_return_code(status)

    def get_simulation_time(self, first_call=False):
        # Return simulation time from signal set on the scene
        return self.get_float_signal(SIMULATION_TIME, first_call=first_call)

    def get_last_cmd_time(self):
        return vrep.simxGetLastCmdTime(self.id) / 1000.0

    def get_ping_time(self):
        status, ping = vrep.simxGetPingTime(self.id)
        self.check_return_code(status)
        return ping

    def get_handle(self, name):
        status, handle = vrep.simxGetObjectHandle(self.id, name, vrep.simx_opmode_blocking)
        self.check_return_code(status)
        return handle

    def read_prox_sensor(self, handle, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, state, coord, _, _ = vrep.simxReadProximitySensor(self.id, handle, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return state, coord

    def get_position(self, handle, first_call=False, relative=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        relative_mode = vrep.sim_handle_parent if relative else -1
        status, pos = vrep.simxGetObjectPosition(self.id, handle, relative_mode, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return pos

    def get_orientation(self, handle, first_call=False, relative=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        relative_mode = vrep.sim_handle_parent if relative else -1
        status, pos = vrep.simxGetObjectOrientation(self.id, handle, relative_mode, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return pos

    def get_velocity(self, handle, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, linear, angular = vrep.simxGetObjectVelocity(self.id, handle, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return linear, angular

    def get_joint_position(self, handle, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, pos = vrep.simxGetJointPosition(self.id, handle, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return pos

    def get_integer_signal(self, signal, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, value = vrep.simxGetIntegerSignal(self.id, signal, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return value

    def get_float_signal(self, signal, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, value = vrep.simxGetFloatSignal(self.id, signal, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return value

    def get_vision_sensor_image(self, handle, first_call=False):
        opmode = vrep.simx_opmode_streaming if first_call else vrep.simx_opmode_buffer
        status, resolution, image = vrep.simxGetVisionSensorImage(self.id, handle, 0, opmode)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
        return resolution, image

    def set_position(self, handle, pos, relative=False):
        relative_mode = vrep.sim_handle_parent if relative else -1
        status = vrep.simxSetObjectPosition(self.id, handle, relative_mode, pos, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)

    def set_orientation(self, handle, ori, relative=False):
        relative_mode = vrep.sim_handle_parent if relative else -1
        status = vrep.simxSetObjectOrientation(self.id, handle, relative_mode, ori, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)

    def set_joint_position(self, handle, pos):
        status = vrep.simxSetJointPosition(self.id, handle, pos, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)

    def set_joint_target_position(self, handle, pos):
        status = vrep.simxSetJointTargetPosition(self.id, handle, pos, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)

    def set_joint_target_velocity(self, handle, vel):
        status = vrep.simxSetJointTargetVelocity(self.id, handle, vel, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)

    def set_vision_sensor_image(self, handle, image):
        status = vrep.simxSetVisionSensorImage(self.id, handle, image, 0, vrep.simx_opmode_oneshot)
        self.check_return_code(status, tolerance=vrep.simx_return_novalue_flag)
