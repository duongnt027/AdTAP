import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class LatentSpaceEnv:
    def __init__(self, device="cpu", max_steps=10, target_sim=0.7):
        self.device = device
        self.max_steps = max_steps
        self.target_sim = target_sim

        self.X = None
        self.Y = None

    def set_data(self, X, Y):
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)

        self.pos_idx = (self.Y == 1).nonzero(as_tuple=True)[0]
        self.neg_idx = (self.Y == 0).nonzero(as_tuple=True)[0]

        assert len(self.pos_idx) > 0, "No label=1 samples"

    def reset(self):
        self.current_idx = self.pos_idx[
            torch.randint(len(self.pos_idx), (1,))
        ].item()

        self.x_orig = self.X[self.current_idx]
        self.x_current = self.x_orig.clone()

        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return self.x_current.clone()

    def compute_reward(self, x_prime):
        pos_samples = self.X[self.pos_idx]
        neg_samples = self.X[self.neg_idx]

        mean_pos = torch.mean(torch.norm(x_prime - pos_samples, dim=1))
        mean_neg = torch.mean(torch.norm(x_prime - neg_samples, dim=1))
        class_margin = torch.sigmoid(mean_neg - mean_pos)

        cos_pos = F.cosine_similarity(
            x_prime.unsqueeze(0), pos_samples, dim=1
        )
        max_sim = torch.max(cos_pos)
        diversity_penalty = torch.relu(max_sim - 0.95)

        sim_orig = F.cosine_similarity(x_prime, self.x_orig, dim=0)
        target_reward = torch.exp(
            -((sim_orig - self.target_sim) ** 2) / 0.02
        )

        reward = (
            0.5 * class_margin
            + 0.4 * target_reward
            - 0.6 * diversity_penalty
        )

        return reward

    def step(self, action):
        self.step_count += 1

        x_prime = self.x_current + action
        reward = self.compute_reward(x_prime)

        self.x_current = x_prime.detach()
        done = self.step_count >= self.max_steps

        return self.get_state(), reward, done

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, s, a, logp, r, v, d):
        self.states.append(s)
        self.actions.append(a)
        self.logps.append(logp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(d)

class PPO:
    def __init__(
        self,
        actor,
        critic,
        env,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        train_epochs=5,
        device="cpu"
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.env = env

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.train_epochs = train_epochs
        self.device = device

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def collect_trajectories(self, num_episodes=4):
        self.buffer.clear()
        episode_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0.0

            for _ in range(self.env.max_steps):
                state = state.to(self.device)

                mu, std = self.actor(state)
                dist = torch.distributions.Normal(mu, std)

                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)

                value = self.critic(state)

                next_state, reward, done = self.env.step(action)

                self.buffer.add(
                    state.detach(),
                    action.detach(),
                    logp.detach(),
                    reward.detach(),
                    value.detach(),
                    done
                )

                state = next_state
                ep_reward += reward.item()

                if done:
                    break

            episode_rewards.append(ep_reward)

        return {"avg_reward": float(np.mean(episode_rewards))}

    def compute_gae(self):
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

            next_value = values[t]

        return torch.stack(advantages), torch.stack(returns)

    def learn(self):
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_logps = torch.stack(self.buffer.logps)

        advantages, returns = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            logps = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(logps - old_logps)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

        avg_reward = torch.stack(self.buffer.rewards).mean().item()
        return actor_loss.item(), critic_loss.item(), avg_reward

    @torch.no_grad()
    def infer(self, z0, steps=None, step_scale=0.1):
        self.actor.eval()

        if steps is None:
            steps = self.env.max_steps

        z = z0.clone().to(self.device)

        for _ in range(steps):
            mu, _ = self.actor(z)
            z = z + step_scale * torch.tanh(mu)

        return z

