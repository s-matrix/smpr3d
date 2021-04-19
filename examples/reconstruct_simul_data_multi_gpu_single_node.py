from smpr3d.util import *
from smpr3d.algorithm import *
from smpr3d.setup import *
import torch as th
import os
import numpy as np

# salloc -C gpu -N 2 -t 30 -c 10 --gres=gpu:8 -A m1759 --ntasks-per-node=8
# srun -N 2 python ./admm_smatrix_dist_pytorch.py
# module purge
# module load pytorch/v1.4.0-gpu
# module list
# Currently Loaded Modulefiles:
#   1) esslurm              2) modules/3.2.11.1     3) cuda/10.1.168        4) nccl/2.5.6
args = Param()

args.io = Param()
args.io.path = '/home/philipp/drop/Public/nesap_hackathon/'
# args.io.path = '../Inputs/'
if os.environ.get('SLURM_PROCID') is not None:
    args.io.path = '/global/cscratch1/sd/pelzphil/'
args.io.summary_log_dir = args.io.path + 'log/'
args.io.logname = 'atoms_aberrations_big'
args.io.filename_data = 'atoms_aberrations_big.h5'

summary = setup_logging(args.io.path, args.io.logname)

args.dist_backend = 'mpi'  # 'mpi'
args.dist_init_method = f'file://{args.io.path}sharedfile'
args.node_config = configure_node(args.dist_backend, args.dist_init_method)

args.beam_threshold_percent = 5e-3
args.max_phase_error = np.pi / 64
args.use_full_smatrix = True
args.uniform_initial_intensity = False

dC1 = 30
# %% load data
i = 0
args.io.filename_results = f'random4_dC{dC1}perc_res_{i + 5:03d}.h5'
world_size = args.node_config.world_size
rank = args.node_config.rank
device = args.node_config.device

lam, alpha_rad, C, dx, specimen_thickness_angstrom, vacuum_probe, D, K, K_rank, MY, MX, NY, NX, \
fy, fx, detector_shape, r, I_target, y_max, x_max, y_min, x_min, S_sol, Psi_sol, r_sol = load_smatrix_data_list2(
    args.io.path + args.io.filename_data, device, rank, world_size, subset=[0, 1, 2, 3])
# dx = 1/2/dx
lam *= 1e10
ss = S_sol.shape
S_sol1 = th.zeros((ss[0],ss[1]+2,ss[2]+2)).to(S_sol.device)
S_sol1[:,:-2,:-2] = S_sol
S_sol = S_sol1
# %% define data-dependent variables
# Fourier space grid on detector

qnp = fourier_coordinates_2D([MY, MX], dx.numpy(), centered=False)
q = th.as_tensor(qnp, device=device)
q2 = th.as_tensor(np.linalg.norm(qnp, axis=0) ** 2, device=device)

# initial aperture amplitude
A_init = initial_probe_amplitude(vacuum_probe, I_target, world_size, rank)

# mask which beams to include in the S-matrix input channels
take_beams = vacuum_probe > args.beam_threshold_percent

B, B_tile, tile_order, beam_numbers, tile_map = prepare_beam_parameters(take_beams, q2, specimen_thickness_angstrom,
                                                                        alpha_rad * 1.1, lam, args.max_phase_error,
                                                                        args.use_full_smatrix, device)
# shape of reconstruction variables
S_shape = (B_tile, NY, NX)
Psi_shape = (D, MY, MX)
z_shape = tuple(I_target.shape)

# map of convergence angles
alpha = q.norm(dim=0) * lam
beam_alphas = th.zeros_like(take_beams, dtype=th.float32, device=device) * -1
beam_alphas[take_beams] = alpha[take_beams]
alpha_map = beam_alphas[take_beams]

# %%
print(specimen_thickness_angstrom)
S0, depth_init = initial_smatrix(S_shape, beam_numbers, device, is_unitary=True, include_plane_waves=B == B_tile,
                                 initial_depth=specimen_thickness_angstrom, lam=lam, q2=q2,
                                 is_pinned=False)

tile_numbers = beam_numbers[beam_numbers >= 0]
beam_numbers = th.ones_like(take_beams).cpu().long() * -1
beam_numbers[take_beams] = th.arange(B)
# %% define S-matrix forward and adjoint operators
from smpr3d.operators import A as A1, AH_S as AH_S1

r_min = th.zeros(2, device=device)


def A(S, Psi, r):
    return A1(S, Psi, r, r_min=r_min, out=None, Mx=MX, My=MY)


def AH_S(S, Psi, r):
    return AH_S1(S, Psi, r, r_min=r_min, out=None, tau=th.tensor([1.0]).to(device), Ny=NY, Nx=NX)


AH_Psi = None
AH_r = None

a = th.sqrt(I_target)

report_smatrix_parameters(rank, world_size, a, S0, B, D, K, MY, MX, NY, NX, fy, fx, B_tile, K_rank,
                          specimen_thickness_angstrom, depth_init, y_max, x_max, y_min, x_min)

if world_size == 1:
    plot(take_beams.cpu().float().numpy(), 'take_beams')
    plot(np.fft.fftshift(beam_numbers.cpu().float().numpy()), 'aperture_tiling', cmap='gist_ncar')
# else:
#     dist.barrier()
# %% define initial probes
C_target = C.to(device)
C_target[1] = 10
print('C_target:', C_target)

C_model = th.zeros(12, D).to(device)
C_model[:] = C_target

#  define data-dependent variables
# Fourier space grid on detector

detector_shape = np.array([MY, MX])
qnp = fourier_coordinates_2D([MY, MX], dx.numpy(), centered=False)
q = th.as_tensor(qnp, device=device)
q2 = th.as_tensor(np.linalg.norm(qnp, axis=0) ** 2, device=device)

# initial aperture amplitude
Ap0 = vacuum_probe
# del I_target
# mask which beams to include in the S-matrix input channels
# take_beams = vacuum_probe > args.beam_threshold_percent / 100

Psi_gen = ZernikeProbe2(q, lam, fft_shifted=True)
Psi_target = Psi_gen(C_target, Ap0).detach()
Psi_model = Psi_gen(C_model, Ap0).detach()
psi_model = th.fft.ifft2(Psi_model, norm='ortho')
cb = fftshift_checkerboard(MY // 2, MX // 2)

fpr1 = Psi_target[0].cpu().numpy()
pr1 = np.fft.ifft2(fpr1, norm='ortho')

fpr2 = Psi_model[0].cpu().numpy()
pr2 = np.fft.ifft2(fpr2, norm='ortho')

from smpr3d.core import SMeta

s_meta = SMeta(take_beams, dx, S_shape, MY, MX, device)
print(s_meta.q_dft)
# report_initial_probes(summary, rank, world_size, Psi_model, psi_model, C_model, specimen_thickness_angstrom, q, lam,
#                       alpha_rad)
# %% perform reconstruction
# m = [MY, MX]
# plotAbsAngle(complex_numpy(S_sol[0, m[0]:-m[0], m[1]:-m[1]].cpu()), f'S_sol[{0}]')
args.reconstruction_opts = Param()
args.reconstruction_opts.max_iters = 100
args.reconstruction_opts.beta = 1.0
args.reconstruction_opts.tau_S = 1e-4
args.reconstruction_opts.tau_Psi = 1e6
args.reconstruction_opts.tau_r = 8e-3
args.reconstruction_opts.optimize_psi = lambda i: i > 1e3
args.reconstruction_opts.node_config = args.node_config
args.reconstruction_opts.verbose = 2

r0 = r
Psi0 = Psi_sol
(S_n, Psi_n, C_n, r_n), outs, opts = fasta2(s_meta, A, AH_S, AH_Psi, AH_r, prox_D_gaussian, Psi_gen, a, S0, Psi0,
                                            C_model, Ap0, r0, args.reconstruction_opts, S_sol=S_sol, Psi_sol=Psi_sol,
                                            r_sol=r_sol, summary=summary)

# save_results(rank, S_n, Psi_n, C_n, r_n, outs, S_sol, Psi_sol, r_sol, beam_numbers, tile_map, alpha_map, A.coords, A.inds,
#              take_beams, lam, alpha_rad, dx, specimen_thickness_angstrom, args.io.path + args.io.filename_results)
# if world_size > 1:
#     dist.barrier()
#     dist.destroy_process_group()
# %%
# plotcx(S_n[2])
