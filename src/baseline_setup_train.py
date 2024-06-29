# This file is designated for the fair comparisions of trajectory balance objective function 
# to the prior work https://github.com/LarsHoldijk/SOCTransitionPaths

import torch
import proxy
import random
from tqdm import tqdm
from utils.utils import pairwise_dist 

import argparse
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger


class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.train:
            self.replay = ReplayBuffer(args)

    def sample(self, args, mds, std):
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device) * std)
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)

        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        mds.set_temperature(0) # We use only external noise.
        for s in tqdm(range(args.num_steps)):
            noise = noises[:, s]
            bias = args.bias_scale * self.policy(position.detach()).squeeze().detach()
            action = bias + noise
            mds.step(1000*action) # 1000 corresponds to global parameter k of the prior work

            position, _, _, potential = mds.report()

            actions[:, s] = action
            positions[:, s+1] = position
            potentials[:, s+1] = potential - (1000*action*position).sum((-2, -1))
        mds.reset()

        log_md_reward = -0.5 * torch.square(actions/args.std).mean((1, 2, 3))
        
        pd = pairwise_dist(position)
        target_pd = pairwise_dist(mds.target_position)
        log_target_reward = -0.5 * torch.square((pd-target_pd)/args.sigma).mean((1, 2))
        
        log_reward = log_md_reward + log_target_reward

        if args.train:
            self.replay.add((positions, actions, log_reward))

        last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device) # For the fair comparisions, we train on reward for fixed length and take final index as 500.
        
        log = {
            'actions': actions,
            'last_idx': last_idx,
            'positions': positions, 
            'potentials': potentials,
            'target_position': mds.target_position,
            'last_position': positions[torch.arange(args.num_samples), last_idx],
        }

        if args.train:
            self.replay.add((positions, actions, log_reward))

        return log

    def train(self, args):
        log_z_optimizer = torch.optim.SGD([self.policy.log_z], lr=args.log_z_lr)
        mlp_optimizer = torch.optim.SGD(self.policy.mlp.parameters(), lr=args.mlp_lr)

        positions, actions, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1])
        
        log_z = self.policy.log_z
        log_forward = -0.5 * torch.square((biases-actions)/args.std).mean((1, 2, 3))
        loss = (log_z+log_forward-log_reward).square().mean() 
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.log_z, args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.mlp.parameters(), args.max_grad_norm)
        
        mlp_optimizer.step()
        log_z_optimizer.step()
        mlp_optimizer.zero_grad()
        log_z_optimizer.zero_grad()
        return loss.item()

class ReplayBuffer:
    def __init__(self, args):
        self.buffer = []
        self.buffer_size = args.buffer_size

    def add(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self):
        idx = random.randrange(len(self.buffer))
        return self.buffer[idx]
    

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--save_dir', default='results/alanine/', type=str)

# Policy Config
parser.add_argument('--force', action='store_true', help='Predict force otherwise potential')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=500, type=int, help='Length of paths')
parser.add_argument('--bias_scale', default=1, type=float, help='Scale factor of bias')
parser.add_argument('--timestep', default=1, type=float, help='Timestep of integrator')
parser.add_argument('--sigma', default=0.05, type=float, help='Control reward of arrival')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=300, type=float, help='Temperature for evaluation')
parser.add_argument('--std', default=0.1, type=float, help='Approximate standard deviation of Langevin integrator used in https://github.com/LarsHoldijk/SOCTransitionPaths')

# Training Config
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')
parser.add_argument('--num_rollouts', default=10000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--log_z_lr', default=1e-2, type=float, help='Learning rate of estimator for log Z')
parser.add_argument('--mlp_lr', default=1e-3, type=float, help='Learning rate of bias potential or force')
parser.add_argument('--buffer_size', default=100, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--end_std', default=0.1, type=float, help='Final standard deviation of annealing schedule')
parser.add_argument('--start_std', default=0.2, type=float, help='Initial standard deviation of annealing schedule')
parser.add_argument('--trains_per_rollout', default=2000, type=int, help='Number of training per rollout in a rollout')

args = parser.parse_args()

if __name__ == '__main__':
    args.train = True

    torch.manual_seed(args.seed)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    logger.info(f"Initialize {args.num_samples} MDs starting at {args.start_state}")
    mds = MDs(args)

    logger.info("Start training")

    annealing_schedule = torch.linspace(args.start_std, args.end_std, args.num_rollouts)
    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')

        std = annealing_schedule[rollout]
        log = agent.sample(args, mds, std)

        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc='Training'):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout

        logger.log(loss, agent.policy, rollout, **log)

    logger.info("Finish training")
    