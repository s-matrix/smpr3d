from smatrix2.default_dependencies import *
from pyms.Probe import wavev
from smatrix2.util import *
from smatrix2.operators.s_matrix.kernel_wrappers import A_fast_full, smatrix_phase_factorsBDK, smatrix_phase_factorsKB, \
    A_fast_full2, A_fast_full3, A_fast_full4
from timeit import default_timer as time

D = 1
K = 64
MY = MX = 64
f = np.array([2, 2])
N_coherent_simulation = np.array([MY, MX]) * f
dx_smatrix_simulation = [1.0, 1.0]
dx_coherent_detector = dx_smatrix_simulation
E = 300e3
kz = wavev(E)
lam = 1 / kz

C1_target = np.linspace(1000, 2000, D, dtype=np.float32)
alpha_aperture_rad = 4e-3
q_aperture = alpha_aperture_rad / lam

dev = th.device('cuda:0')

q_detector = fourier_coordinates_2D([MY, MX], dx_smatrix_simulation)
# S-matrix lateral dimensions

qy0 = fftfreq(MY, dx_smatrix_simulation[0])
qx0 = fftfreq(MX, dx_smatrix_simulation[1])
# Fourier space grid on detector
qt = th.from_numpy(q_detector).float()

C1_target_th = th.from_numpy(C1_target)
A_init = aperture(qt, lam, alpha_aperture_rad, edge=0)
probe = ZernikeProbe(qt, lam, A_init, A_requires_grad=True, fft_shifted=True, C1=C1_target_th)
Psi_target = probe()
Psi = Psi_target.detach().to(dev)
# plot(A_init)
# %%
r = th.zeros((D, K, 2)).long().to(dev)
for d in range(D):
    r[d, :, 0] = 14.5

xv = fftfreq(N_coherent_simulation[0], 1 / N_coherent_simulation[0])
yv = fftfreq(N_coherent_simulation[1], 1 / N_coherent_simulation[1])
[xa, ya] = np.meshgrid(np.round(yv), np.round(xv))

q_coherent_simulation = fourier_coordinates_2D(N_coherent_simulation, dx_coherent_detector)
q2_coherent_simulation = np.linalg.norm(q_coherent_simulation, axis=0) ** 2
beam_mask_coherent_simulation = (q2_coherent_simulation < np.max(q_aperture) ** 2) * (ya % f[0] == 0) * (xa % f[1] == 0)
beam_mask_coherent_simulation = th.from_numpy(beam_mask_coherent_simulation).bool()
B = th.sum(beam_mask_coherent_simulation).item()
beam_numbers_coherent_simulation = th.ones(tuple(N_coherent_simulation)) * -1
beam_numbers_coherent_simulation[beam_mask_coherent_simulation] = th.arange(B).float()

# plot(q2_coherent_simulation, 'q2_coherent_simulation')
# plot(beam_numbers_coherent_simulation.cpu(), 'beam_numbers_coherent_simulation')
# %%
qx, qy = np.meshgrid(fftfreq(MX), fftfreq(MY))
q = np.array([qy, qx])
coords = fftshift(np.array(np.mgrid[-MY // 2:MY // 2, -MX // 2:MX // 2]), (1, 2))
inds = np.mgrid[:MY, :MX]

beam_numbers1 = beam_numbers_coherent_simulation[::f[0], ::f[1]].to(dev)
take_beams = beam_mask_coherent_simulation[::f[0], ::f[1]].to(dev)
q1 = th.from_numpy(q).to(dev)
q_indices1 = th.from_numpy(coords).to(dev)
tile_map = th.arange(B).long().to(dev)

out = th.zeros((B, D, K, 2)).to(dev)
out2 = th.zeros((K, B, 2)).to(dev)
# (Psi, r, take_beams, q, q_indices, B, out=None)
phase_factors = smatrix_phase_factorsBDK(Psi, r, take_beams, q1, q_indices1, B, out)
phase_factors2 = smatrix_phase_factorsKB(Psi[0], r[0], take_beams, q1, q_indices1, B, out2)

S = th.ones((B, *N_coherent_simulation, 2)).to(dev)
for b in range(B):
    cur_beam = beam_numbers_coherent_simulation == b
    cur_planewave = make_real(cur_beam)
    S[b] = th.ifft(cur_planewave, 2, True)
# %%
out = th.zeros((D, K, MY, MX, 2)).to(dev)
# S, phase_factors, r, r_min, out=None, Mx=0, My=0
r_min = th.tensor([0, 0]).to(dev)

for i in range(3):
    start1 = time()
    exitw = A_fast_full(S, phase_factors, r, r_min, out, Mx=MX, My=MY)
    th.cuda.synchronize(dev)
    end1 = time()

for i in range(3):
    exitw2 = A_fast_full2(S, phase_factors2, r[0], r_min, MY, MX)
    start2 = time()
    exitw2 = A_fast_full2(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end2 = time()

for i in range(3):
    exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
    start3 = time()
    exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end3 = time()

for i in range(3):
    out2 = th.zeros((D, K, MY, MX, 2)).to(dev)
    exitw4 = A_fast_full4(S, phase_factors, r, r_min, out2, MY, MX)
    start4 = time()
    exitw4 = A_fast_full4(S, phase_factors, r, r_min, out2, MY, MX)
    th.cuda.synchronize(dev)
    end4 = time()

print(f'A_fast_full (custom kernel)           {end1 - start1}')
print(f'A_fast_full4 (custom kernel 2)           {end4 - start4}')
print(f'A_fast_full2 (real batched matmul)    {end2 - start2}')
print(f'A_fast_full3 (complex batched matmul) {end3 - start3}')
cb = fftshift_checkerboard(MX // 2, MY // 2)
# %%
i = 0
rexitw = th.ifft(exitw[:, i], 2, True)
# rexitw2 = th.ifft(exitw2, 2, True)
#%%

#%%
# plotcxmosaic(fftshift(complex_numpy(rexitw2.cpu()) * cb))
# plotcxmosaic(complex_numpy(exitw2.cpu()))
# #%%
# plotAbsAngle(complex_numpy(exitw[0,0].cpu())-complex_numpy(exitw2[0].cpu()))
# #%%
# plotAbsAngle(complex_numpy(exitw[0,0].cpu())-complex_numpy(exitw3[0].cpu()))
