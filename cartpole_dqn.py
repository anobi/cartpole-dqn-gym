import sys
import gym
import torch

from cartpole_dqn.dqn import DQN
from cartpole_dqn.trainer import CartPoleDQNTrainer
from cartpole_dqn.runner import CartPoleRunner
from cartpole_dqn.utils.device import DeviceUse, get_device_family

SEED = 3907
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
IMAGE_SIZE = 80


class CartPoleDQN:
    def __init__(self):
        self.device = torch.device(device=get_device_family(DeviceUse.DEVICE))
        self.rng = torch.Generator(device=get_device_family(DeviceUse.RNG))
        self.gym_env =  gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.action_space = self.gym_env.action_space.n
        self.image_size = IMAGE_SIZE
        self.screen_width = 0
        self.screen_height = 0

        self.rng.manual_seed(SEED)
        self.net = self.init_net()
    
    def init_net(self):
        self.gym_env.reset(seed=SEED)
        self.screen_height, self.screen_width, _ = self.render().shape
        return DQN(self.device, self.image_size, self.image_size, self.action_space).to(self.device)

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def render(self):
        return self.gym_env.render()

    def reset(self):
        return self.gym_env.reset()

    def step(self, *args):
        return self.gym_env.step(*args)

    def close(self):
        self.gym_env.close()


def print_usage():
    print("Usage: python cartpole_v1.py [train | run] [episodes (train) | {pretrained model file} (run)]")


def main(argv):
    if len(argv) < 2:
        print_usage()
        exit()

    mode = argv[0]
    if mode not in ['train', 'run']:
        print_usage()

    agent = CartPoleDQN()

    if mode == 'train':
        episodes = int(argv[1])
        print(f"Training on {agent.device}")
        trainer = CartPoleDQNTrainer(
            agent,
            agent.device,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
            target_update=TARGET_UPDATE,
            w=agent.image_size,
            h=agent.image_size,
            action_space=agent.action_space,
            memory=MEMORY_SIZE
        )
        trainer.set_state_file(f'CartPoleDQN_{episodes}ep.pt')
        trainer.run(episodes, plot=True)

    elif mode == 'run':
        model_file = argv[1]
        agent.load_net(model_file)

        print(f"Running on {agent.device}")
        runner = CartPoleRunner(agent)
        runner.run()

    agent.render()
    agent.close()


if __name__ == "__main__":
    main(sys.argv[1:])
