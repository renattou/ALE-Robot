import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "y", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser()

    envarg = parser.add_argument_group("Environment")
    envarg.add_argument("game", help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym.")
    envarg.add_argument("--environment", choices=["ale", "gym", "robot"], default="robot", help="Whether to train agent using ALE, OpenAI Gym or Robot+ALE.")
    envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")
    envarg.add_argument("--display_processed_screen", type=str2bool, default=False, help="Display game screen after preprocessing during training and testing.")
    envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
    envarg.add_argument("--repeat_action_probability", type=float, default=0, help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
    envarg.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True, help="Use minimal action set.")
    envarg.add_argument("--color_averaging", type=str2bool, default=True, help="Perform color averaging with previous frame.")
    envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
    envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
    envarg.add_argument("--record_screen_path", help="Record game screens under this path. Subfolder for each game is created.")
    envarg.add_argument("--record_sound_filename", help="Record game sound in this file.")
    envarg.add_argument("--game_difficulty", type=int, default=0, help="Set game difficulty on ALE.")
    envarg.add_argument("--game_mode", type=int, default=0, help="Set game mode on ALE.")

    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
    memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--enable_double", type=str2bool, default=True, help="Use Double DQN.")
    netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
    netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
    netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
    netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
    netarg.add_argument("--optimizer_decay", type=float, default=0.95, help="Decay rate (rho) for RMSProp and Adadelta algorithms.")
    netarg.add_argument("--optimizer_epsilon", type=float, default=0.01, help="Epsilon for RMSProp, Adam and Adadelta algorithms.")
    netarg.add_argument('--loss', choices=['mse', 'huber'], default='mse', help='Network loss function.')
    #netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
    #netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
    #netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
    #netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start", type=float, default=1.0, help="Exploration rate at the beginning of decay.")
    antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
    antarg.add_argument("--exploration_decay_steps", type=float, default=2000000, help="How many steps to decay the exploration rate.")
    antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
    antarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
    antarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
    antarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many game steps.")
    antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
    mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
    mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
    mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
    mainarg.add_argument("--start_epoch", type=int, default=0, help="Start from this epoch, affects exploration rate and names of saved snapshots.")
    mainarg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")
    mainarg.add_argument("--load_weights", help="Load network from file (with path and extension).")
    mainarg.add_argument("--save_weights_prefix", help="Save network to given file (with path and no extension). Epoch and extension will be appended.")
    mainarg.add_argument("--csv_file", help="Write training progress to this file (with path).")

    twoarg = parser.add_argument_group("Two-Player")
    twoarg.add_argument("--two_player", type=str2bool, default=False, help="Enable two player mode on modified ALE.")
    twoarg.add_argument("--pygame_width", type=int, default=640, help="Screen width of the pygame screen.")
    envarg.add_argument("--pygame_height", type=int, default=768, help="Screen height of the pygame screen.")

    simvarg = parser.add_argument_group("V-REP simulation")
    simvarg.add_argument("--ip", default="127.0.0.1", help="Simulation IP used by V-REP.")
    simvarg.add_argument("--port", type=int, default=-25000, help="Simulation port used by V-REP.")
    simvarg.add_argument("--display_camera", type=str2bool, default=False, help="Display camera from the simulation.")
    simvarg.add_argument("--camera_screen_scale", type=int, default=1, help="Scale factor from the original camera resolution.")
    simvarg.add_argument("--wait_action", type=str2bool, default=False, help="Wait for action to be executed by robot before next agent step.")
    simvarg.add_argument("--enable_sync", type=str2bool, default=True, help="Enable synchronous communication between client and V-REP. This will force target_fps = -1.")

    comarg = parser.add_argument_group("Common")
    comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
    comarg.add_argument("--target_fps", type=int, default=60, help="Target FPS for the experiment. Use -1 for max FPS.")

    args = parser.parse_args()
    return args

def huber_loss(y_true, y_pred):
    from keras import backend as K
    # From https://github.com/matthiasplappert/keras-rl/blob/master/rl/util.py
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    clip_value = 1
    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))
