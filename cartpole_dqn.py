import sys
import gym
import math
import random
import torch

import torchvision.transforms as T
from cartpole_dqn.dqn import DQN
from cartpole_dqn.trainer import CartPoleDQNTrainer
from cartpole_dqn.runner import CartPoleRunner
from cartpole_dqn.utils.screen import get_torch_screen, get_human_screen

SEED = 3907
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10
IMAGE_SIZE = 84


class CartPoleDQN:
    def __init__(self):
        self.device = self._get_device()
        self.rng = self._get_rng_generator()
        self.env =  gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.action_space = self.env.action_space.n
        self.image_size = IMAGE_SIZE
        self.screen_width = 0
        self.screen_height = 0
        self.resize = T.Compose([T.ToPILImage(), T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])

        self.rng.manual_seed(SEED)
        self.net = self.init_net()

    def _get_device(self):
        device_family = 'cpu'
        if torch.cuda.is_available() and torch.backends.cuda.is_built():
            device_family = 'cuda'
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_family = 'mps'
        
        return torch.device(device_family)

    def _get_rng_generator(self):
        device_family = 'cpu'
        if torch.cuda.is_available() and torch.backends.cuda.is_built():
            device_family = 'cuda'

        return torch.Generator(device=device_family)
    
    def init_net(self):
        self.env.reset(seed=SEED)
        self.screen_height, self.screen_width, _ = self.render().shape
        return DQN(self.device, self.image_size, self.image_size, self.action_space).to(self.device)

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def select_action(self, state, steps_done):
        sample = torch.rand(1, generator=self.rng, device=self.device).item()
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

    env = CartPoleDQN()

    if mode == 'train':
        episodes = int(argv[1])
        print(f"Training on {env.device}")
        trainer = CartPoleDQNTrainer(
            env,
            env.device,
            batch_size=BATCH_SIZE, 
            gamma=GAMMA, 
            eps_start=EPS_START,
            eps_end=EPS_END, 
            eps_decay=EPS_DECAY, 
            target_update=TARGET_UPDATE,
            w=env.image_size,
            h=env.image_size,
            action_space=env.action_space
        )
        trainer.set_state_file(f'CartPoleDQN_{episodes}ep.pt')
        trainer.run(episodes)

    elif mode == 'run':
        model_file = argv[1]
        env.load_net(model_file)

        print(f"Running on {env.device}")
        runner = CartPoleRunner(env)
        runner.run()

    env.render()
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
