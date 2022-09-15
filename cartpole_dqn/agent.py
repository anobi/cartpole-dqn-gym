from email.generator import Generator
import torch
from gym import Env
from typing import NamedTuple
from cartpole_dqn.dqn import DQN
from cartpole_dqn.trainer import CartPoleDQNTrainer
from cartpole_dqn.runner import CartPoleRunner


class Hyperparameters(NamedTuple):
    BATCH_SIZE: int
    TARGET_UPDATE: int
    MEMORY_SIZE: int
    GAMMA: float
    EPSILON_START: float
    EPSILON_END: float
    EPSILON_DECAY: float


class CartPoleDQNAgent:
    def __init__(self, env: Env, device: torch.device, image_size: int, rng: Generator):
        self.environment = env
        self.action_space = env.action_space.n
        self.device = device
        self.rng = rng
        self.image_size = image_size
        self.screen_width = 0
        self.screen_height = 0
        self.policy_net = self.init_net()
    
    def init_net(self):
        self.screen_height, self.screen_width, _ = self.render().shape
        return DQN(self.device, self.image_size, self.image_size, self.action_space).to(self.device)

    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

    def render(self):
        return self.environment.render()

    def reset(self):
        return self.environment.reset()

    def step(self, *args):
        return self.environment.step(*args)

    def close(self):
        self.environment.close()

    def run(self, episodes):
        runner = CartPoleRunner(self)
        runner.run(episodes)

    def train(self, episodes: int, state_file: str, params: Hyperparameters):
        trainer = CartPoleDQNTrainer(
            self,
            self.device,
            batch_size=params.BATCH_SIZE,
            gamma=params.GAMMA,
            eps_start=params.EPSILON_START,
            eps_end=params.EPSILON_END,
            eps_decay=params.EPSILON_DECAY,
            target_update=params.TARGET_UPDATE,
            w=self.image_size,
            h=self.image_size,
            action_space=self.action_space,
            memory=params.MEMORY_SIZE,
            state_file=state_file
        )
        trainer.set_state_file(state_file)
        trainer.run(episodes, plot=False)