from smpr3d.util import *
from smpr3d.algorithm import *
from smpr3d.setup import *
from smpr3d.torch_imports import *
import os

# salloc -C gpu -N 2 -t 30 -c 10 --gres=gpu:8 -A m1759 --ntasks-per-node=8
# srun -N 2 python ./admm_smatrix_dist_pytorch.py
# module purge
# module load pytorch/v1.4.0-gpu
# module list
# Currently Loaded Modulefiles:
#   1) esslurm              2) modules/3.2.11.1     3) cuda/10.1.168        4) nccl/2.5.6
args = Param()

args.io = Param()
args.io.path = '/mnt/4TB/projects/misc/smatrix_paper/'
# args.io.path = '../Inputs/'
if os.environ.get('SLURM_PROCID') is not None:
    args.io.path = '/global/cscratch1/sd/pelzphil/'
args.io.summary_log_dir = args.io.path + 'log/'
args.io.logname = 'atoms_aberrations24'
args.io.filename_data = 'atoms_aberrations_big.h5'

summary = setup_logging(args.io.path, args.io.logname, args.io.summary_log_dir)

args.dist_backend = 'mpi'  # 'mpi'
args.dist_init_method = f'file://{args.io.path}sharedfile'
args.node_config = configure_node(args.dist_backend, args.dist_init_method)

args.beam_threshold_percent = 5e-3
args.max_phase_error = np.pi / 64
args.use_full_smatrix = True
args.uniform_initial_intensity = False

dC1 = 30
#%% load data
args.io.filename_results = f'random_big.h5'
world_size = args.node_config.world_size
rank = args.node_config.rank
device = args.node_config.device

lam, alpha_rad, C, dx, specimen_thickness_angstrom, vacuum_probe, D, K, K_rank, MY, MX, NY, NX, \
fy, fx, detector_shape, r, I_target, y_max, x_max, y_min, x_min, S_sol, Psi_sol, r_sol = load_smatrix_data_list2(
    args.io.path + args.io.filename_data, device, rank, world_size, subset=[0,1,2,3])
# dx = 1/2/dx
lam *= 1e10
# %% define data-dependent variables
# Fourier space grid on detector

qnp = fourier_coordinates_2D([MY, MX], dx.numpy(), centered=False)
q = th.as_tensor(qnp, device=device)
q2 = th.as_tensor(np.linalg.norm(qnp, axis=0) ** 2, device=device)

# initial aperture amplitude
A_init = initial_probe_amplitude(vacuum_probe, I_target, world_size, rank, summary)

# mask which beams to include in the S-matrix input channels
take_beams = vacuum_probe > args.beam_threshold_percent

B, B_tile, tile_order, beam_numbers, tile_map = prepare_beam_parameters(take_beams, q2, specimen_thickness_angstrom,
                                                                        alpha_rad * 1.1, lam, args.max_phase_error,
                                                                        args.use_full_smatrix, device)
# shape of reconstruction variables
S_shape = (B_tile, NY, NX, 2)
Psi_shape = (D, MY, MX, 2)
z_shape = tuple(I_target.shape + (2,))

# map of convergence angles
alpha = q.norm(dim=0) * lam
beam_alphas = th.zeros_like(take_beams, dtype=th.float32, device=device) * -1
beam_alphas[take_beams] = alpha[take_beams]
alpha_map = beam_alphas[take_beams]
# %%
# plot(tile_order.cpu())
# bn = beam_numbers.cpu().numpy()
# plot(bn)

# %%
# print(specimen_thickness_angstrom)
S0, depth_init = initial_smatrix(S_shape, beam_numbers, device, is_unitary=True, include_plane_waves=B == B_tile,
                                 initial_depth=specimen_thickness_angstrom, lam=lam, q2=q2, dtype=th.float32,
                                 is_pinned=False)

# plotcx(complex_numpy(depth_init.cpu()))

tile_numbers = beam_numbers[beam_numbers >= 0]
beam_numbers = th.ones_like(take_beams).cpu().long() * -1
beam_numbers[take_beams] = th.arange(B)
# %% define S-matrix forward and adjoint operators
# forward operator for S-matrix
mode = 'fast'  # or 'low_memory'
A = SMatrixExitWave(S_shape, z_shape, device, detector_shape, B, beam_numbers, tile_map, mode=mode)
# adjoint operator for S-matrix
AH_S = A.H

# adjoint operator for probe shape_in, shape_out, device, detector_shape, B, beam_numbers, tile_map, r_min=None, mode='fast'
AH_Psi = SMatrixExitWaveAdjointProbe(A.oshape, Psi_shape, device, detector_shape, B, beam_numbers, tile_map)
# TODO: adjoint operator for positions
AH_r = None

# Fourier transform
F = FFT2(A.oshape)
# inverse/adjoint Fourier transform
FH = F.H

a = th.sqrt(I_target)

report_smatrix_parameters(rank, world_size, a, S0, B, D, K, MY, MX, NY, NX, fy, fx, B_tile, K_rank,
                          specimen_thickness_angstrom, depth_init, y_max, x_max, y_min, x_min)

if world_size == 1:
    plot(take_beams.cpu().float().numpy(), 'take_beams')
    plot(fftshift(beam_numbers.cpu().float().numpy()), 'aperture_tiling', cmap='gist_ncar')
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
take_beams = vacuum_probe > args.beam_threshold_percent / 100

Psi_gen = ZernikeProbe2(q, lam, fft_shifted=True)
Psi_target = Psi_gen(C_target, Ap0).detach()
Psi_model = Psi_gen(C_model, Ap0).detach()
psi_model = th.ifft(Psi_model, 2, True)
cb = fftshift_checkerboard(MY // 2, MX // 2)

fpr1 = complex_numpy(Psi_target[0].cpu())
pr1 = np.fft.ifft2(fpr1, norm='ortho')

fpr2 = complex_numpy(Psi_model[0].cpu())
pr2 = np.fft.ifft2(fpr2, norm='ortho')

# plotcx(np.hstack([fpr1, fpr2]), 'fpr')
# plotcx(np.hstack([pr1, pr2]), 'pr')

# report_initial_probes(summary, rank, world_size, Psi_model, psi_model, C_model, specimen_thickness_angstrom, q, lam,
#                       alpha_rad)
# %% perform reconstruction
# m = [MY, MX]
args.reconstruction_opts = Param()
args.reconstruction_opts.max_iters = 120
args.reconstruction_opts.beta = 1.0
args.reconstruction_opts.tau_S = 1e-3
args.reconstruction_opts.tau_Psi = 1e6
args.reconstruction_opts.tau_r = 8e-3
args.reconstruction_opts.optimize_psi = lambda i: i > 1e3
args.reconstruction_opts.node_config = args.node_config
args.reconstruction_opts.verbose = 2

r0 = r
Psi0 = Psi_sol
(S_n, Psi_n, C_n, r_n), outs, opts = fasta2(A, AH_S, AH_Psi, AH_r, F, FH, prox_D_gaussian, Psi_gen, a, S0, Psi0, C_model,
                                       Ap0, r0, args.reconstruction_opts, S_sol=S_sol, Psi_sol=Psi_sol, r_sol=r_sol,
                                       summary=summary)


# save_results(rank, S_n, Psi_n, C_n, r_n, outs, S_sol, Psi_sol, r_sol, beam_numbers, tile_map, alpha_map, A.coords, A.inds,
#              take_beams, lam, alpha_rad, dx, specimen_thickness_angstrom, args.io.path + args.io.filename_results)
# if world_size > 1:
#     dist.barrier()
#     dist.destroy_process_group()
# %%
# plotcx(S_n[2])
