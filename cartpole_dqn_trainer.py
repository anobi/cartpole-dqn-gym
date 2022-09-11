import torch
import torch.optim as optim

from tqdm import trange
from itertools import count
from cartpole_dqn import DQN, optimize_model
from cartpole_screen import get_torch_screen, capture_frames
from plotting import Plotter
from memory import Memory


class CartPoleDQNTrainer:
    def __init__(self, env, batch_size=128, gamma=0.999, eps_start=0.9,
                 eps_end=0.05, eps_decay=200, target_update=10):
        self.env = env

        self.target_net = DQN(env.screen_height, env.screen_width, env.action_space).to(env.device)
        self.optimizer = optim.RMSprop(self.env.net.parameters())
        self.memory = Memory(100000)

        self.steps_done = 0
        self.total_reward = 0
        self.episode_durations = []
        self.episode_rewards = []

        self.plotter = Plotter("Training", 3)

    def run(self, num_episodes=50, batch_size=128, gamma=0.999, target_update=10, frames_per_state=4, plot=False):

        self.target_net.load_state_dict(self.env.net.state_dict())
        self.target_net.eval()

        for ie in (tr := trange(num_episodes)):
            # Reset the environment and fetch a new initial state screenshot
            self.env.env.reset()
            episode_reward = 0

            frame = self.env.render()
            last_state = None
            current_state = get_torch_screen(frame, self.env.device, self.env.image_size, self.env.resize)

            for t in count():
                action = self.env.select_action(current_state, self.steps_done)
                _, reward, done, _, _ = self.env.env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device=self.env.device)

                # Observe new state
                last_state = current_state
                current_frame = self.env.render()
                current_state = get_torch_screen(current_frame, self.env.device, self.env.image_size, self.env.resize)

                # Store transition and to replay memory and perform gradient descent
                self.memory.push(last_state, action, current_state, torch.tensor([episode_reward], device=self.env.device))

                optimize_model(self.memory, batch_size, self.optimizer,
                               self.env.net, self.target_net, gamma,
                               self.env.device)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(episode_reward)

                    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

                    if len(durations_t) >= 100:
                        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                        means = torch.cat((torch.zeros(99), means))
                        tr.set_description("mean duration %.2f" % torch.mean(means))

                    if plot:
                        self.plotter.set("durations", self.episode_durations)
                        self.plotter.set("rewards", self.episode_rewards)
                        self.plotter.set("means", means)
                        self.plotter.draw()
                    break

                if ie % target_update == 0:
                    self.target_net.load_state_dict(self.env.net.state_dict())

        torch.save(self.env.net.state_dict(), f'CartPoleDQN_{num_episodes}ep.pt')
