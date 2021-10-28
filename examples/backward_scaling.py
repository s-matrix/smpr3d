from builtins import complex

from smatrix2.default_dependencies import fourier_coordinates_2D
from smatrix2.operators.s_matrix.SMatrixExitWave import SMatrixExitWave
from smatrix2.util import *
import os
from timeit import default_timer as time

logFormatter = logging.Formatter("%(asctime)s %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
D = 2

dx = np.array([0.2, 0.2])
E = 300e3
lam = wavelength(E)
C1_target = np.linspace(0, 100, D, dtype=np.float32)
alpha_rad = 20e-3
q_aperture = alpha_rad / lam
dtype = np.float32

args = Param()

args.beam_threshold_percent = 5
args.max_phase_error = np.pi / 200
args.use_full_smatrix = True
args.uniform_initial_intensity = True

# %% load data
world_size = 1
rank = 0
device = th.device('cuda:0')

C = th.zeros(12, D).to(device)
specimen_thickness_angstrom = 100
r0 = None
a = None
I_target = None
Psi0 = None
y_max, x_max, y_min, x_min = 0, 0, 0, 0

skip_nyquist_positions = 0.75
dx_scan = dx * skip_nyquist_positions
FOV_simulation = th.as_tensor(np.array([70, 70]) * dx_scan)
scan_shape = th.round(((FOV_simulation / dx_scan))).int() // 2
# create positions
K = int(th.prod(scan_shape))
K_rank = K
pos = th.zeros((K, 2))
ls = np.linspace(0, scan_shape[0] * skip_nyquist_positions, scan_shape[0], endpoint=True, dtype=np.float32)
pos[:, 0] = th.from_numpy(np.repeat(ls, scan_shape[1])) * skip_nyquist_positions
ls = np.linspace(0, scan_shape[1] * skip_nyquist_positions, scan_shape[1], endpoint=True, dtype=np.float32)
pos[:, 1] = th.from_numpy(np.tile(ls.reshape(1, scan_shape[1]), scan_shape[0])) * skip_nyquist_positions

r0 = np.tile(pos[None, ...], (D, 1, 1))
r0 = th.from_numpy(r0).to(device)

C1_model = C[0]
# C1_model[1] = 10
# C1_model[2] = 20
C12a = C[1]
# C12a[:] = 1e-8
C12b = C[2]
# C12b[:] = 1e-8
C21a = C[3]
# C21a[:] = 1e-8
C21b = C[4]
C23a = C[5]
C23b = C[6]
C3 = C[7]
C32a = C[8]
C32b = C[9]
C34a = C[10]
C34b = C[11]

# %% define data-dependent variables
# Fourier space grid on detector
s = 30
e = 160
Ms = np.linspace(s,e,int((e-s)/10)+1,endpoint=True, dtype=np.int)
times = np.zeros(Ms.shape)
Bs = np.zeros(Ms.shape)

MY = MX = Ms[0]
fx, fy = 2, 2
NX, NY = MX * fx, MY * fy
detector_shape = np.array([MY, MX])
qnp = fourier_coordinates_2D([MY, MX], dx, centered=False)
q = th.as_tensor(qnp, device=device)
q2 = th.as_tensor(np.linalg.norm(qnp, axis=0) ** 2, device=device)

# initial aperture amplitude
A_init = (th.sqrt(q2) < alpha_rad / lam).float()
vacuum_probe = A_init

# del I_target
# mask which beams to include in the S-matrix input channels
take_beams = vacuum_probe > args.beam_threshold_percent / 100

probe = ZernikeProbe(q, lam, A_init, A_requires_grad=True, fft_shifted=True, C1=C1_model, C12a=C12a, C12b=C12b,
                     C21a=C21a, C21b=C21b, C23a=C23a, C23b=C23b, C3=C3, C32a=C32a, C32b=C32b, C34a=C34a, C34b=C34b)
Psi_init1 = probe().detach()

B, B_tile, tile_order, tile_number, tile_map = prepare_beam_parameters(take_beams, q2, specimen_thickness_angstrom,
                                                               alpha_rad, lam, args.max_phase_error,
                                                               args.use_full_smatrix, device)
# shape of reconstruction variables
S_shape = (B_tile, NY, NX, 2)
Psi_shape = (D, MY, MX, 2)
z_shape = (D, K, MY, MX, 2)
S0, depth_init = initial_smatrix(S_shape, tile_number, device, is_unitary=True, include_plane_waves=B == B_tile,
                                 initial_depth=specimen_thickness_angstrom, lam=lam, q2=q2, dtype=th.float32,
                                 is_pinned=False)

# forward operator for S-matrix
mode = 'fast'  # or 'low_memory'
A = SMatrixExitWave(S_shape, z_shape, device, detector_shape, B, tile_number, tile_map, mode=mode)
# adjoint operator for S-matrix
AH_S = A.H

start = time()
z = th.zeros((z_shape), device=device, dtype=th.float32)
S = AH_S(z, Psi_init1, r0)
th.cuda.synchronize(device)
end = time()

del S0
del z
th.cuda.empty_cache()

inds = th.randperm(K)

r0 = r0[:, inds, :].contiguous()

for i, M in enumerate(Ms):
    MY = MX = M
    fx, fy = 3, 3
    NX, NY = MX * fx, MY * fy
    detector_shape = np.array([MY, MX])
    qnp = fourier_coordinates_2D([MY, MX], dx, centered=False)
    q = th.as_tensor(qnp, device=device)
    q2 = th.as_tensor(np.linalg.norm(qnp, axis=0) ** 2, device=device)

    # initial aperture amplitude
    A_init = (th.sqrt(q2) < alpha_rad / lam).float()
    vacuum_probe = A_init

    # del I_target
    # mask which beams to include in the S-matrix input channels
    take_beams = vacuum_probe > args.beam_threshold_percent / 100

    probe = ZernikeProbe(q, lam, A_init, A_requires_grad=True, fft_shifted=True, C1=C1_model, C12a=C12a, C12b=C12b,
                         C21a=C21a, C21b=C21b, C23a=C23a, C23b=C23b, C3=C3, C32a=C32a, C32b=C32b, C34a=C34a, C34b=C34b)
    Psi_init1 = probe().detach()

    B, B_tile, tile_order, tile_number, tile_map = prepare_beam_parameters(take_beams, q2, specimen_thickness_angstrom,
                                                                   alpha_rad, lam, args.max_phase_error,
                                                                   args.use_full_smatrix, device)
    # shape of reconstruction variables
    S_shape = (B_tile, NY, NX, 2)
    Psi_shape = (D, MY, MX, 2)
    z_shape = (D, K, MY, MX, 2)
    S0, depth_init = initial_smatrix(S_shape, tile_number, device, is_unitary=True, include_plane_waves=B == B_tile,
                                     initial_depth=specimen_thickness_angstrom, lam=lam, q2=q2, dtype=th.float32,
                                     is_pinned=False)

    # forward operator for S-matrix
    mode = 'fast'  # or 'low_memory'
    A = SMatrixExitWave(S_shape, z_shape, device, detector_shape, B, tile_number, tile_map, mode=mode)
    # adjoint operator for S-matrix
    AH_S = A.H

    start = time()
    z = th.zeros((z_shape), device=device, dtype=th.float32)
    print(f'B_tile = {B_tile}   z shape: {z_shape}')
    S = AH_S(z, Psi_init1, r0)
    th.cuda.synchronize(device)
    end = time()

    times[i] = (end - start)
    Bs[i] = B_tile

    print(f'{i:03d} M={M} time = {times[i]}')

    del S0
    del z
    th.cuda.empty_cache()



#%%
batch_size = 16
threadsperblock = 512

pow = 4
pow2 = 2
fac = 1e7
fac2 = 10e2
f, a = plt.subplots()
a.loglog(Ms,times, label='execution time')
a.loglog(Ms,Ms**pow/fac, label=f'M^{pow}/{fac:2.2g}')
a.loglog(Ms,Ms**pow2/fac2, label=f'M^{pow2}/{fac:2.2g}')
a.set_xlabel('M')
a.set_ylabel('time [s]')
plt.legend()
plt.title(f'D={D} K={K} batch size {batch_size} threadsperblock {threadsperblock}')
# a.loglog(Ms,times**1)
plt.show()
f.savefig('backward_scaling.pdf')

#%%
# f, a = plt.subplots()
# a.loglog(Ms,Bs)
# a.loglog(Ms,Ms**2)
# plt.show()