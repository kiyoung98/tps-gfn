import torch
from .utils import *
import openmm.unit as unit
from torch.distributions import Normal

class Metric:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, dtype=torch.float, device=args.device).unsqueeze(0)
        self.std = torch.tensor(md.std.value_in_unit(unit.nanometer/unit.femtosecond), dtype=torch.float, device=args.device)

        self.normal = Normal(0, self.std)

        if args.molecule == 'alanine':
            self.angle_2 = torch.tensor([1, 6, 8, 14], dtype=torch.long, device=args.device)
            self.angle_1 = torch.tensor([6, 8, 14, 16], dtype=torch.long, device=args.device)
        elif args.molecule == 'histidine':
            self.angle_2 = torch.tensor([0, 6, 8, 11], dtype=torch.long, device=args.device)
            self.angle_1 = torch.tensor([6, 8, 11, 23], dtype=torch.long, device=args.device)

    def expected_pairwise_distance(self, last_position, target_position):
        last_pd = pairwise_dist(last_position)
        target_pd = pairwise_dist(target_position)
        
        pd = (last_pd-target_pd).square().mean((1, 2))
        mean_pd, std_pd = pd.mean().item(), pd.std().item()
        return mean_pd, std_pd

    def expected_pairwise_coulomb_distance(self, last_position, target_position):
        last_pcd = self.coulomb(last_position)
        target_pcd = self.coulomb(target_position)
        
        pcd = (last_pcd-target_pcd).square().mean((1, 2))
        mean_pcd, std_pcd = pcd.mean().item(), pcd.std().item()
        return mean_pcd, std_pcd

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye # eye prevents 0 division
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix

    def log_likelihood(self, diff):
        log_likelihood = self.normal.log_prob(diff).sum((2, 3)).mean(1)
        mean_ll, std_ll = log_likelihood.mean().item(), log_likelihood.std().item()
        return mean_ll, std_ll

    def cv_metrics(self, positions, target_position, potentials):
        etps, efps, etp_idxs, efp_idxs = [], [], [], []

        target_psi = compute_dihedral(target_position[:, self.angle_1].unsqueeze(0))
        target_phi = compute_dihedral(target_position[:, self.angle_2].unsqueeze(0))

        psi = compute_dihedral(positions[:, :, self.angle_1])
        phi = compute_dihedral(positions[:, :, self.angle_2])

        hit_mask = (torch.abs(psi-target_psi) < 0.75) & (torch.abs(phi-target_phi) < 0.75)
        hit, hit_idxs = hit_mask.max(-1)

        thp = 100 * hit.sum().float() / hit.shape[0]

        for i, hit_idx in enumerate(hit_idxs):
            if hit_idx > 0:
                etp, idx = potentials[i][:hit_idx].max(0)
                etps.append(etp)
                etp_idxs.append(idx.item())

                efp = potentials[i][hit_idx]
                efps.append(efp)
                efp_idxs.append(hit_idx.item())

        if len(etps)>0:
            etps = torch.tensor(etps)
            efps = torch.tensor(efps)

            mean_etp = etps.mean().item()
            mean_efp = efps.mean().item()

            std_etp = etps.std().item()
            std_efp = efps.std().item()

            return thp, etps, etp_idxs, mean_etp, std_etp, efps, efp_idxs, mean_efp, std_efp
        else:
            return thp, None, None, None, None, None, None, None, None