import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_dihedral(p): 
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])

    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)

class AlaninePotential():
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./src/utils/alanine.dat"

        with open(file) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            # if line == '  \n':
            #     psi = psi + 1
            #     phi = 0
            #     continue
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z
    

class HistidinePotential(): # TODO: Make histidine.dat for 4 torsion angles
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./src/utils/histidine.dat"

        with open(file) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            # if line == '  \n':
            #     psi = psi + 1
            #     phi = 0
            #     continue
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

def plot_paths_alanine(save_dir, rollout, positions, target_position, last_idx):
    positions = positions.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    potential = AlaninePotential()
    xs = np.arange(-np.pi, np.pi + .1, .1)
    ys = np.arange(-np.pi, np.pi + .1, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    plt.contourf(xs, ys, z, levels=100, zorder=0)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / positions.shape[0]) for i in range(positions.shape[0])])

    psis_start = []
    phis_start = []

    for i in range(positions.shape[0]):
        psis_start.append(compute_dihedral(positions[i, 0, angle_1, :]))
        phis_start.append(compute_dihedral(positions[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(last_idx[i]):
            psi.append(compute_dihedral(positions[i, j, angle_1, :]))
            phi.append(compute_dihedral(positions[i, j, angle_2, :]))
        ax.plot(phi, psi, marker='o', linestyle='None', markersize=2, alpha=1.)

    ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')
    
    psis_target = []
    phis_target = []
    psis_target.append(compute_dihedral(target_position[0, angle_1, :]))
    phis_target.append(compute_dihedral(target_position[0, angle_2, :]))
    ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.show()
    plt.savefig(f'{save_dir}/paths/{rollout}.png')
    plt.close()
    return fig


def plot_paths_histidine(save_dir, rollout, positions, target_position, last_idx): # TODO: Two Ramachandran Plots
    positions = positions.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    angle_2 = [0, 6, 8, 11]
    angle_1 = [6, 8, 11, 23]

    potential = HistidinePotential()
    xs = np.arange(-np.pi, np.pi + .1, .1)
    ys = np.arange(-np.pi, np.pi + .1, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    plt.contourf(xs, ys, z, levels=100, zorder=0)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / positions.shape[0]) for i in range(positions.shape[0])])

    psis_start = []
    phis_start = []

    for i in range(positions.shape[0]):
        psis_start.append(compute_dihedral(positions[i, 0, angle_1, :]))
        phis_start.append(compute_dihedral(positions[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(last_idx[i]):
            psi.append(compute_dihedral(positions[i, j, angle_1, :]))
            phi.append(compute_dihedral(positions[i, j, angle_2, :]))
        ax.plot(phi, psi, marker='o', linestyle='None', markersize=2, alpha=1.)

    ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')
    
    psis_target = []
    phis_target = []
    psis_target.append(compute_dihedral(target_position[0, angle_1, :]))
    phis_target.append(compute_dihedral(target_position[0, angle_2, :]))
    ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.show()
    plt.savefig(f'{save_dir}/paths/{rollout}.png')
    plt.close()
    return fig

def plot_potentials(save_dir, rollout, potentials, last_idx):
    potentials = potentials.detach().cpu().numpy()
    fig = plt.figure(figsize=(20, 5))
    for i in range(potentials.shape[0]):
        if last_idx[i] > 0:
            plt.plot(potentials[i][:last_idx[i]])
    
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f'{save_dir}/potentials/{rollout}.png')
    plt.close()
    return fig

def plot_etps(save_dir, rollout, etps, etp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(etp_idxs, etps)
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f'{save_dir}/etps/{rollout}.png')
    plt.close()
    return fig

def plot_efps(save_dir, rollout, efps, efp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(efp_idxs, efps)
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f'{save_dir}/efps/{rollout}.png')
    plt.close()
    return fig

def plot_path_alanine(save_dir, positions, target_position, last_idx):
    positions = positions.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()

    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]
    
    for i in range(positions.shape[0]):
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        potential = AlaninePotential()
        xs = np.arange(-np.pi, np.pi + .1, .1)
        ys = np.arange(-np.pi, np.pi + .1, .1)
        x, y = np.meshgrid(xs, ys)
        inp = torch.tensor(np.array([x, y])).view(2, -1).T

        z = potential.potential(inp)
        z = z.view(y.shape[0], y.shape[1])

        plt.contourf(xs, ys, z, levels=100, zorder=0)
        
        psis_start = []
        phis_start = []

        psis_start.append(compute_dihedral(positions[i, 0, angle_1, :]))
        phis_start.append(compute_dihedral(positions[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(last_idx[i]):
            psi.append(compute_dihedral(positions[i, j, angle_1, :]))
            phi.append(compute_dihedral(positions[i, j, angle_2, :]))
        ax.plot(phi, psi, marker='o', linestyle='None', markersize=2, alpha=1.)

        ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')
        
        psis_target = []
        phis_target = []
        psis_target.append(compute_dihedral(target_position[0, angle_1, :]))
        phis_target.append(compute_dihedral(target_position[0, angle_2, :]))
        ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.show()
        plt.savefig(f'{save_dir}/path/{i}.png')
        plt.close() 
        
def plot_3D_view(save_dir, start_file, positions, last_idx):
    positions = positions.detach().cpu().numpy()
    for i in tqdm(range(positions.shape[0]), desc='Plot 3D views'):
        if last_idx[i] > 0:
            for j in range(last_idx[i]):
                traj = md.load_pdb(start_file)
                traj.xyz = positions[i, j]
                
                if j == 0:
                    trajs = traj
                else:
                    trajs = trajs.join(traj)
            trajs.save(f'{save_dir}/3D_views/{i}.h5')
    
def plot_potential(save_dir, potentials, last_idx):
    potentials = potentials.detach().cpu().numpy()
    for i in tqdm(range(potentials.shape[0]), desc='Plot potentials'):
        if last_idx[i] > 0:
            plt.figure(figsize=(16, 2))
            plt.plot(potentials[i][:last_idx[i]])
            plt.xlabel('Time (fs)')
            plt.ylabel("Potential Energy (kJ/mol)")
            plt.show()
            plt.savefig(f'{save_dir}/potential/{i}.png')
            plt.close()
