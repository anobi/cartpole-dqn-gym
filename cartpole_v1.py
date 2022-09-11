import sys
import gym
import math
import random
import torch
import pygame

import torchvision.transforms as T
from cartpole_dqn import DQN
from cartpole_dqn_trainer import CartPoleDQNTrainer
from cartpole_runner import CartPoleRunner
from cartpole_screen import get_torch_screen, get_human_screen

SEED = 3907
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10
IMAGE_SIZE = 84

cuda_device = torch.device('cuda')
RNG = torch.Generator(device='cuda')
RNG.manual_seed(SEED)

pygame.init()


class CartPoleV1:
    def __init__(self, device):
        self.device = device
        self.env =  gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.action_space = self.env.action_space.n

        self.net = None
        self.image_size = IMAGE_SIZE
        self.screen_width = 0
        self.screen_height = 0
        self.resize = T.Compose([T.ToPILImage(), T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])

        self.init_net()

    def init_net(self):
        self.env.reset(seed=SEED)
        _, self.screen_height, self.screen_width = self.render().shape

        self.net = DQN(self.image_size, self.image_size, self.action_space).to(self.device)

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def select_action(self, state, steps_done):
        sample = torch.rand(1, generator=RNG, device=cuda_device).item()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def print_usage():
    print("Usage: python cartpole_v1.py [train | run] [episodes (train) | {pretrained model file} (run)]")


def main(argv):
    if len(argv) < 2:
        print_usage()
        exit()

    mode = argv[0]
    if mode not in ['train', 'run']:
        print_usage()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = CartPoleV1(device)

    if mode == 'train':
        episodes = int(argv[1])
        print(f"Training on {device}")
        trainer = CartPoleDQNTrainer(env, batch_size=BATCH_SIZE, gamma=GAMMA, eps_start=EPS_START,
                                     eps_end=EPS_END, eps_decay=EPS_DECAY, target_update=TARGET_UPDATE)
        trainer.run(episodes)

    elif mode == 'run':
        model_file = argv[1]
        env.load_net(model_file)

        print(f"Running on {device}")
        runner = CartPoleRunner(env)
        runner.run()

    env.render()
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
