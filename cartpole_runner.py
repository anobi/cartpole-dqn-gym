import numpy as np
from tqdm import trange
from itertools import count
from cartpole_screen import get_screen
from gym.utils.play import display_arr

import pygame

class CartPoleRunner:
    def __init__(self, env):
        self.env = env
        self.steps_done = 0
        self.episode_durations = []

        self.display = pygame.display.set_mode((800, 600))

    def run(self, num_episodes=50):
        for _ in (tr := trange(num_episodes)):
            self.env.env.reset()

            last_screen = get_screen(self.env.env, self.env.device, monochrome=True)
            current_screen = get_screen(self.env.env, self.env.device, monochrome=True)
            state = current_screen - last_screen

            display_arr(self.display, self.env.env.render(), [600, 800], True)

            for t in count():
                action = self.env.select_action(state, self.steps_done)
                _, reward, done, _, _ = self.env.env.step(action.item())

                last_screen = current_screen
                current_screen = get_screen(self.env.env, self.env.device, monochrome=True)
                if done:
                    next_state = None
                else:
                    next_state = current_screen - last_screen
                state = next_state

                if done:
                    self.episode_durations.append(t + 1)
                    tr.set_description("min: %.2f, max: %.2f, mean: %.2f" % (
                        np.min(self.episode_durations),
                        np.max(self.episode_durations),
                        np.mean(self.episode_durations)
                    ))
                    break
