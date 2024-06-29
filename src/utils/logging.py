import os
import sys
import wandb
import torch
import logging

from .plot import *
from .metrics import Metric

class Logger():
    def __init__(self, args, md):
        self.type = args.type
        self.wandb = args.wandb
        self.molecule = args.molecule
        self.start_file = md.start_file
        self.save_freq = args.save_freq if args.type == 'train' else 1

        self.best_loss = float('inf')
        self.metric = Metric(args, md)

        self.save_dir = os.path.join(args.save_dir, args.project, args.date, args.type, str(args.seed))
   
        for name in ['paths', 'path', 'potentials', 'potential', 'etps', 'efps', 'policies', '3D_views']:
            if not os.path.exists(f'{self.save_dir}/{name}'):
                os.makedirs(f'{self.save_dir}/{name}')

        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = args.type + '.log'
        log_file = os.path.join(self.save_dir, log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
            
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        for k, v in vars(args).items():
            self.logger.info(f"{k}: {v}")

    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def log(
            self, 
            loss, 
            policy, 
            rollout, 
            actions,
            last_idx,
            positions, 
            potentials, 
            last_position,
            target_position,
        ):

        # Calculate metrics
        if self.molecule in ['alanine', 'histidine']:
            thp, etps, etp_idxs, etp, std_etp, efps, efp_idxs, efp, std_efp = self.metric.cv_metrics(positions, target_position, potentials)

        ll, std_ll = self.metric.log_likelihood(actions)
        pd, std_pd = self.metric.expected_pairwise_distance(last_position, target_position)
        pcd, std_pcd = self.metric.expected_pairwise_coulomb_distance(last_position, target_position)
        len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        # Log
        if self.type == 'train':
            self.logger.info("-----------------------------------------------------------")
            self.logger.info(f'Rollout: {rollout}')
            self.logger.info(f"loss: {loss}")
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(policy.state_dict(), f'{self.save_dir}/policy.pt')

        if self.wandb:
            log = {
                    'loss': loss,
                    'log_z': policy.log_z.item(),
                    'll': ll,
                    'epd': pd,
                    'epcd': pcd,
                    'len': len,
                    'std_ll': std_ll,
                    'std_pd': std_pd,
                    'std_pcd': std_pcd,
                    'std_len': std_len,
                }

            if self.molecule in ['alanine', 'histidine']:
                cv_log = {
                        'thp': thp,
                        'etp': etp,
                        'efp': efp,
                        'std_etp': std_etp,
                        'std_efp': std_efp,
                    }
                log.update(cv_log)

            wandb.log(log, step=rollout)

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"ll: {ll}")
        self.logger.info(f"epd: {pd}")
        self.logger.info(f"epcd: {pcd}")
        self.logger.info(f"len: {len}")
        self.logger.info(f"std_ll: {std_ll}")
        self.logger.info(f"std_pd: {std_pd}")
        self.logger.info(f"std_pcd: {std_pcd}")
        self.logger.info(f"std_len: {std_len}")

        if self.molecule in ['alanine', 'histidine']:
            self.logger.info(f"thp: {thp}")
            self.logger.info(f"etp: {etp}")
            self.logger.info(f"efp: {efp}")
            self.logger.info(f"std_etp: {std_etp}")
            self.logger.info(f"std_etp: {std_efp}")

        if rollout % self.save_freq == 0:
            torch.save(policy.state_dict(), f'{self.save_dir}/policies/{rollout}.pt')

            if self.molecule == 'alanine':
                fig_path = plot_paths_alanine(self.save_dir, rollout, positions, target_position, last_idx)
            elif self.molecule == 'histidine':
                fig_path = plot_paths_histidine(self.save_dir, rollout, positions, target_position, last_idx)

            fig_pot = plot_potentials(self.save_dir, rollout, potentials, last_idx)

            if self.wandb:
                log = {'potentials': wandb.Image(fig_pot)}
            
                if self.molecule in ['alanine', 'histidine']:
                    fig_etp = plot_etps(self.save_dir, rollout, etps, etp_idxs)
                    fig_efp = plot_efps(self.save_dir, rollout, efps, efp_idxs) 
                    
                    cv_log = {
                        'paths': wandb.Image(fig_path),
                        'etps': wandb.Image(fig_etp),
                        'efps': wandb.Image(fig_efp),
                        }
                    log.update(cv_log)
            
                wandb.log(log, step=rollout)
            
        if self.type == 'eval':
            plot_path_alanine(self.save_dir, positions, target_position, last_idx)
            plot_potential(self.save_dir, potentials, last_idx)
            plot_3D_view(self.save_dir, self.start_file, positions, last_idx)