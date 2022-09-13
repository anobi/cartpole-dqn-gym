import gym
import torch

from cartpole_dqn.dqn import DQN
from cartpole_dqn.utils.device import DeviceUse, get_device_family


RNG_SEED = 3907


class CartPoleDQNAgent:
    def __init__(self, image_size):
        self.device = torch.device(device=get_device_family(DeviceUse.DEVICE))
        self.rng = torch.Generator(device=get_device_family(DeviceUse.RNG))
        self.gym_env =  gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.action_space = self.gym_env.action_space.n
        self.image_size = image_size
        self.screen_width = 0
        self.screen_height = 0

        self.rng.manual_seed(RNG_SEED)
        self.net = self.init_net()
    
    def init_net(self):
        self.gym_env.reset(seed=RNG_SEED)
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