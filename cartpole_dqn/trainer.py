import math
import random
import torch
import torch.optim as optim
import numpy as np

from tqdm import trange
from itertools import count

from cartpole_dqn.dqn import DQN, optimize_model
from cartpole_dqn.utils.screen import get_torch_screen, capture_frames
from cartpole_dqn.utils.plotting import Plotter
from cartpole_dqn.utils.memory import Memory


class CartPoleDQNTrainer:
    def __init__(self, env, device, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, w, h, action_space, memory):
        self.env = env
        self.device = device
        self.image_size = w
        self.target_net = DQN(device, w, h, action_space).to(device)
        self.optimizer = optim.RMSprop(self.env.net.parameters())
        self.memory = Memory(memory)

        self.action_space=action_space
        self.batch_size=batch_size
        self.gamma=gamma
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.target_update=target_update

        self.steps_done = 0
        self.total_reward = 0

        self.plotter = Plotter("Training", 2)

    def set_state_file(self, path):
        pass

    def select_action(self, state):
        sample = torch.rand(1, generator=self.env.rng, device=self.device).item()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.env.net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)

    def run(self, num_episodes=50, plot=False):
        self.target_net.load_state_dict(self.env.net.state_dict())
        self.target_net.eval()
        losses = []
        rewards = []

        for episode in (tr := trange(num_episodes)):
            # Reset the environment and fetch a new initial state screenshot
            self.env.reset()
            frame = self.env.render()
            current_screen = get_torch_screen(frame, self.device, self.image_size)
            last_screen = get_torch_screen(frame, self.device, self.image_size)
            state = current_screen - last_screen

            episode_reward = 0
            episode_losses = []
            for step in count():
                # Act
                action = self.select_action(state)
                _, reward, done, _, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward.item()
                
                # Observe new state
                last_screen = current_screen
                current_screen = get_torch_screen(self.env.render(), self.device, self.image_size)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store transition and to replay memory and perform gradient descent
                self.memory.push(state, action, next_state, reward)
                state = next_state

                loss = optimize_model(
                    self.memory, 
                    self.batch_size, 
                    self.optimizer,
                    self.env.net, 
                    self.target_net, 
                    self.gamma,
                    self.device
                )
                episode_losses.append(loss or 0.0)

                # Update the target net with the state from the policy net
                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.env.net.state_dict())

                if done:
                    break

            losses.append(np.average(episode_losses))
            avg_loss = np.average(losses)
            rewards.append(episode_reward)
            avg_reward = np.average(rewards)
            tr.set_description(f"loss: {avg_loss:.2f} | reward: {avg_reward:.2f}")
            if plot:
                self.plotter.push(episode, reward=avg_reward, loss=avg_loss)
                self.plotter.draw()

        torch.save(self.env.net.state_dict(), f'CartPoleDQN_{num_episodes}ep.pt')
