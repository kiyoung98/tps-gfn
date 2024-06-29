import os
import wandb
import torch
import argparse

from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--type', default='eval', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)

# Logger Config
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--model_path', default='', type=str)
parser.add_argument('--project', default='alanine', type=str)
parser.add_argument('--save_dir', default='results', type=str)
parser.add_argument('--date', default="date", type=str, help='Date of the training')

# Policy Config
parser.add_argument('--force', action='store_true', help='Network predicts force')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=1000, type=int, help='Length of paths')
parser.add_argument('--bias_scale', default=0.01, type=float, help='Scale factor of bias')
parser.add_argument('--timestep', default=1, type=float, help='Timestep of integrator')
parser.add_argument('--sigma', default=0.05, type=float, help='Control reward of arrival')
parser.add_argument('--num_samples', default=64, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=300, type=float, help='Temperature for evaluation')

args = parser.parse_args()

if __name__ == '__main__':
    if args.wandb:
        wandb.init(project=args.project+'_eval', config=args)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    logger.info(f"Initialize {args.num_samples} MDs starting at {args.start_state}")
    mds = MDs(args)

    model_path = args.model_path if args.model_path else os.path.join(args.save_dir, args.project, args.date, 'train', str(args.seed), 'policy.pt')
    agent.policy.load_state_dict(torch.load(model_path))
    
    logger.info(f"Start Evaulation")
    log = agent.sample(args, mds, args.temperature)
    logger.log(None, agent.policy, 0, **log)
    logger.info(f"Finish Evaluation")