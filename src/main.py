import sys
import random
import utils
from agent import Agent
from environment import ALEEnvironment, GymEnvironment, RobotEnvironment
from dqn import DQN
from playertwo import PlayerTwo
from statistics import Statistics

def main():
    # Process arguments
    args = utils.parse_args()

    # Use random seed from argument
    if args.random_seed:
        random.seed(args.random_seed)

    # Instantiate environment class
    if args.environment == "ale":
        env = ALEEnvironment(args.game, args)
    elif args.environment == "gym":
        env = GymEnvironment(args.game, args)
    elif args.environment == "robot":
        env = RobotEnvironment(args.game, args)
    else:
        assert False, "Unknown environment" + args.environment

    # Instantiate DQN
    action_dim = env.action_dim()
    state_dim = env.state_dim()
    net = DQN(state_dim, action_dim, args)

    # Load weights before starting training
    if args.load_weights:
        filepath = args.load_weights
        net.load(filepath)

    # Instantiate agent
    agent = Agent(env, net, args)

    # Start statistics
    stats = Statistics(agent, agent.net, agent.net.memory, env, args)

    # Play game with two players (user and agent)
    if args.two_player:
        player_b = PlayerTwo(args)
        env.set_mode('test')
        stats.reset()
        agent.play_two_players(player_b)
        stats.write(0, "2player")
        sys.exit()

    # Play agent
    if args.play_games > 0:
        env.set_mode('test')
        stats.reset()
        for _ in range(args.play_games):
            agent.play()
        stats.write(0, "play")
        sys.exit()

    # Populate replay memory with random steps
    if args.random_steps:
        env.set_mode('test')
        stats.reset()
        agent.play_random(args.random_steps)
        stats.write(0, "random")

    for epoch in range(args.start_epoch, args.epochs):
        # Train agent
        if args.train_steps:
            env.set_mode('train')
            stats.reset()
            agent.train(args.train_steps)
            stats.write(epoch + 1, "train")

            # Save weights after every epoch
            if args.save_weights_prefix:
                filepath = args.save_weights_prefix + "_%d.h5" % (epoch + 1)
                net.save(filepath)

        # Test agent
        if args.test_steps:
            env.set_mode('test')
            stats.reset()
            agent.test(args.test_steps)
            stats.write(epoch + 1, "test")

    # Stop statistics
    stats.close()

if __name__ == "__main__":
    main()
