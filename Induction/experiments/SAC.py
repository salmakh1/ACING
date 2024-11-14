
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 1024)
        self.fc2 = nn.Linear(1024, 265)
        self.mean = nn.Linear(265, dim)
        self.log_std = nn.Linear(265, dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        z = normal.rsample()

        action = torch.tanh(z)
        action = (action + 1) / 2
        # log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = normal.log_prob(z) - torch.log(0.5 * (1 - action.pow(2)) + 1e-6)

        return action, log_prob.sum(dim=1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, action):
        x = torch.relu(self.fc1(action))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACAgent:
    def __init__(self, alpha=0.99, dim=10, llm=None, seed=0):
        # torch.cuda.manual_seed((seed+1)*55)
        # torch.cuda.manual_seed_all((seed+1)*55)
        print("Dimension is ", dim)
        self.policy_net = PolicyNetwork(dim).to(device)
        self.q_net1 = QNetwork(dim).to(device)
        self.q_net2 = QNetwork(dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=3e-4)
        self.alpha = alpha  # The hyperparameter responsible for the tradeoff between exploration and exploitation
        self.target_entropy = -float(dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=9e-4)
        self.llm = llm

        self.q_loss = []
        self.policy_loss = []
        self.alpha_loss = []
        self.q_value = []
        self.entropy = []
        self.muc = []

    def update(self, state, action, reward):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor([reward]).to(device)

        with torch.no_grad():
            target_q = reward.item()  # Simplified for single state SAC

        current_q1 = self.q_net1(action)
        print("current_q1 ", current_q1)
        current_q2 = self.q_net2(action)
        print("current_q2 ", current_q2)

        q_loss1 = torch.mean((current_q1 - target_q).pow(2))
        q_loss2 = torch.mean((current_q2 - target_q).pow(2))
        print("loss q network ", q_loss1.item())
        self.q_loss.append(q_loss1.item())
        print("loss q2 network", q_loss2.item())

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        new_action, log_prob = self.policy_net.sample(state=state)
        q1_new = self.q_net1(new_action)
        q2_new = self.q_net2(new_action)
        print("q1_new ", q1_new)
        print("q2_new ", q2_new)

        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()
        self.entropy.append(log_prob.item())
        print("entropy", log_prob.item())
        print("policy_loss", policy_loss.item())
        self.policy_loss.append(policy_loss.item())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        self.alpha = self.alpha.detach()
        print("alpha ", self.alpha)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, _ = self.policy_net.sample(state)
        return action.cpu().detach().numpy()


class SAC():
    def __init__(self, dim, bounds, max_iter, W_LLM, L=1.0):
        print("********Initiating SAC with LIPO ************")
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.active_arms = []
        self.rewards = {}
        self.values = []
        self.counts = {}
        self.W_LLM = W_LLM
        self.L = L  # Lipschitz constant

    def reward_function(self, x, n_evaluations=1):
        x = np.array(x, dtype=np.float32)
        x_projected = self.W_LLM.linear(torch.tensor(x))
        r, _, b = self.W_LLM.eval(x_projected)
        print(f"The average reward after {n_evaluations} evaluations is {r}")
        return r, b

    def run_sac(self):
        agent = SACAgent(dim=self.bounds.shape[0], llm=self.W_LLM)
        state_tensor = torch.FloatTensor(torch.tensor([1.0])).unsqueeze(0)

        t = 0
        count = 0
        while t < int(self.max_iter):
            print(f"We are in iteration {t}")

            # Select action
            action = agent.select_action(state_tensor)
            print("action", action)

            reward, b = self.reward_function(action)
            if not b:
                t += 1

            action_tuple = tuple(action[0]) if isinstance(action, np.ndarray) else tuple(action)
            self.rewards[action_tuple] = reward
            self.active_arms.append(action_tuple)
            self.values.append(reward)
            # Update the agent with the observed transition
            agent.update(state_tensor, action, reward)

        best_index = np.argmax([self.rewards[arm] for arm in self.active_arms])
        best_arm = self.active_arms[best_index]
        best_reward = self.rewards[best_arm]
        return np.array(best_arm), best_reward
