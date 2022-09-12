import torch
import torch.optim as optim

from tqdm import trange
from itertools import count

from cartpole_dqn.dqn import DQN, optimize_model
from cartpole_dqn.utils.screen import get_torch_screen, capture_frames
from cartpole_dqn.utils.plotting import Plotter
from cartpole_dqn.utils.memory import Memory


class CartPoleDQNTrainer:
    def __init__(self, env, device, batch_size=128, gamma=0.999, eps_start=0.9,
                 eps_end=0.05, eps_decay=200, target_update=10, w=84, h=84, action_space=2):
        self.env = env
        self.device = device
        self.image_size = w
        self.target_net = DQN(device, w, h, action_space).to(device)
        self.optimizer = optim.RMSprop(self.env.net.parameters())
        self.memory = Memory(100000)

        self.steps_done = 0
        self.total_reward = 0
        self.episode_durations = []
        self.episode_rewards = []

        self.plotter = Plotter("Training", 3)

    def set_state_file(self, path): 
        pass

    def run(self, num_episodes=50, batch_size=128, gamma=0.999, target_update=10, frames_per_state=4, plot=False):
        self.target_net.load_state_dict(self.env.net.state_dict())
        self.target_net.eval()

        for ie in (tr := trange(num_episodes)):
            # Reset the environment and fetch a new initial state screenshot
            self.env.env.reset()
            frame = self.env.render()
            current_screen = get_torch_screen(frame, self.device, self.image_size, self.env.resize)
            last_screen = get_torch_screen(frame, self.device, self.image_size, self.env.resize)
            state = current_screen - last_screen

            for t in count():
                # Act
                action = self.env.select_action(state, self.steps_done)
                _, reward, done, _, _ = self.env.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_torch_screen(self.env.render(), self.device, self.image_size, self.env.resize)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store transition and to replay memory and perform gradient descent
                self.memory.push(state, action, next_state, reward)
                state = next_state

                optimize_model(
                    self.memory, 
                    batch_size, 
                    self.optimizer,
                    self.env.net, 
                    self.target_net, 
                    gamma,
                    self.device
                )

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(reward)
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

                # Update the target net with the state from the policy net
                if ie % target_update == 0:
                    self.target_net.load_state_dict(self.env.net.state_dict())

        torch.save(self.env.net.state_dict(), f'CartPoleDQN_{num_episodes}ep.pt')