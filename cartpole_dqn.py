import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import Transition


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride, padding=0):
            return ((size - kernel_size + (2 * padding)) // stride) + 1

        conv_w = conv2d_size_out(w, kernel_size=3, stride=1)
        conv_h = conv2d_size_out(h, kernel_size=3, stride=1)
        linear_input_size = conv_w * conv_h * 64

        self.hidden1 = nn.Linear(7 * 7 * 64, 128)

        self.head = nn.Linear(128, outputs)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.size(), x.shape)
        x = F.relu(self.hidden1(x.view(x.size(0), -1)))
        return self.head(x)


def optimize_model(memory, batch_size, optimizer, policy_net, target_net, gamma, device):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Compute MSE loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
