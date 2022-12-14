import pygame
import torch
import numpy as np
from tqdm import trange
from itertools import count
from gym.utils.play import display_arr

from cartpole_dqn.utils.screen import get_torch_screen

class CartPoleRunner:
    def __init__(self, agent):
        self.agent = agent
        self.steps_done = 0
        self.episode_durations = []

        pygame.init()
        pygame.display.set_caption("CartPoleDQN Runner")
        self.display = pygame.display.set_mode((agent.screen_width, agent.screen_height))

    def draw(self, frame):
        display_arr(self.display, frame, [self.agent.screen_width, self.agent.screen_height], True)
        pygame.display.flip()

    def select_action(self, state):
        with torch.no_grad():
            return self.agent.policy_net(state.to(self.agent.device)).max(1)[1].view(1, 1)

    def run(self, num_episodes=50):
        for _ in (tr := trange(num_episodes)):
            self.agent.reset()
            frame = self.agent.render()
            current_screen = get_torch_screen(frame, self.agent.device, self.agent.image_size)
            state = current_screen
            self.draw(frame)

            for t in count():
                action = self.select_action(state)
                _, _, done, _, _ = self.agent.step(action.item())

                current_frame = self.agent.render()
                current_screen = get_torch_screen(current_frame, self.agent.device, self.agent.image_size)
                self.draw(current_frame)
                
                if done:
                    next_state = None
                else:
                    next_state = current_screen # - last_screen
                state = next_state

                if done:
                    self.episode_durations.append(t + 1)
                    tr.set_description("min: %.2f, max: %.2f, mean: %.2f" % (
                        np.min(self.episode_durations),
                        np.max(self.episode_durations),
                        np.mean(self.episode_durations)
                    ))
                    break
