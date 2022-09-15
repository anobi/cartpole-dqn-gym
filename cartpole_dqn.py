import sys
import gym
import torch

from cartpole_dqn.utils.device import DeviceUse, get_device_family
from cartpole_dqn.agent import CartPoleDQNAgent, Hyperparameters


RNG_SEED = 3907
IMAGE_SIZE = 80
PARAMS = Hyperparameters(
    BATCH_SIZE = 128,
    GAMMA = 0.999,
    EPSILON_START = 0.9,
    EPSILON_END = 0.01,
    EPSILON_DECAY = 500,
    TARGET_UPDATE = 10,
    MEMORY_SIZE = 10000
)


def print_usage():
    print("Usage: python cartpole_v1.py [train | run] [episodes] [model file]]")


def main(argv):
    if len(argv) < 3:
        print_usage()
        exit()

    mode = argv[0]
    if mode not in ['train', 'run']:
        print_usage()

    episodes = int(argv[1])
    model_file = argv[2]

    device = torch.device(device=get_device_family(DeviceUse.DEVICE))
    rng = torch.Generator(device=get_device_family(DeviceUse.RNG))
    rng.manual_seed(RNG_SEED)

    environment =  gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
    environment.reset(seed=RNG_SEED)

    agent = CartPoleDQNAgent(environment, device, IMAGE_SIZE, rng)

    if mode == 'train':
        print(f"Training on {agent.device}")
        agent.train(episodes, model_file, PARAMS)
    elif mode == 'run':
        print(f"Running on {agent.device}")
        agent.load_net(model_file)
        agent.run(episodes)

    agent.render()
    agent.close()


if __name__ == "__main__":
    main(sys.argv[1:])
