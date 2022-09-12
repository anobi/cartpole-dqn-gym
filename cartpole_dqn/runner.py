import pygame
import numpy as np
from tqdm import trange
from itertools import count
from gym.utils.play import display_arr

from cartpole_dqn.utils.screen import get_torch_screen, get_human_screen


class CartPoleRunner:
    def __init__(self, env):
        self.env = env
        self.steps_done = 0
        self.episode_durations = []

        pygame.init()
        pygame.display.set_caption("CartPoleDQN Runner")
        self.display = pygame.display.set_mode((env.screen_width, env.screen_height))

    def draw(self, frame):
        display_arr(self.display, frame, [self.env.screen_width, self.env.screen_height], True)
        pygame.display.flip()

    def run(self, num_episodes=50):
        for _ in (tr := trange(num_episodes)):
            self.env.env.reset()
            frame = self.env.render()
            current_screen = get_torch_screen(frame, self.env.device, self.env.image_size, self.env.resize)
            last_screen = current_screen
            state = current_screen
            self.draw(frame)

            for t in count():
                action = self.env.select_action(state, self.steps_done)
                _, _, done, _, _ = self.env.env.step(action.item())

                last_screen = current_screen
                current_frame = self.env.render()
                current_screen = get_torch_screen(current_frame, self.env.device, self.env.image_size, self.env.resize)
                self.draw(current_frame)
                
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
