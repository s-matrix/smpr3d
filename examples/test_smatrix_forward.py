from smpr3d.util import *
from smpr3d.setup import *
from smpr3d.operators import *
from numpy.fft import fftfreq, fftshift
import numpy as np
import torch as th

#f4dstem

from timeit import default_timer as time

D = 1
K = 144
MY = MX = 32
f = np.array([2, 2])
N_coherent_simulation = np.array([MY, MX]) * f
dx_smatrix_simulation = [1.0, 1.0]
dx_coherent_detector = dx_smatrix_simulation
E = 300e3
lam = wavelength(E)

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
phase_factors = smatrix_phase_factorsBDK(Psi, r, take_beams, q1, B, out)
phase_factors2 = smatrix_phase_factorsKB(Psi[0], r[0], take_beams, q1, q_indices1, B, out2)
phase_factors2 = th.view_as_complex(phase_factors2)
Psi.phase_factors = phase_factors

S = th.ones((B, *N_coherent_simulation), dtype=th.complex64).to(dev)
for b in range(B):
    cur_beam = beam_numbers_coherent_simulation == b
    cur_planewave = cur_beam
    S[b] = th.fft.ifft2(cur_planewave, norm='ortho')
# %%
out = th.zeros((D, K, MY, MX, 2)).to(dev)
# S, phase_factors, r, r_min, out=None, Mx=0, My=0
r_min = th.tensor([0, 0]).to(dev)

from smpr3d.operators import A as A1, AH_S as AH_S1

r_min = th.zeros(2, device=dev)

exitw = A1(S, Psi, r, r_min, out, Mx=MX, My=MY)
for i in range(1):
    start1 = time()
    exitw = A1(S, Psi, r, r_min, out, Mx=MX, My=MY)
    th.cuda.synchronize(dev)
    end1 = time()

th.backends.cuda.matmul.allow_tf32 = True
exitw2 = A_fast_full2(S, th.view_as_real(phase_factors2), r[0], r_min, MY, MX)
for i in range(1):
    start2 = time()
    exitw2 = A_fast_full2(S, th.view_as_real(phase_factors2), r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end2 = time()

th.backends.cuda.matmul.allow_tf32 = False
exitw2 = A_fast_full2(S, th.view_as_real(phase_factors2), r[0], r_min, MY, MX)
for i in range(1):
    start6 = time()
    exitw2 = A_fast_full2(S, th.view_as_real(phase_factors2), r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end6 = time()

th.backends.cuda.matmul.allow_tf32 = False
exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
for i in range(1):
    start3 = time()
    exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end3 = time()

th.backends.cuda.matmul.allow_tf32 = True
exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
for i in range(1):
    start5 = time()
    exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end5 = time()

out2 = th.zeros((D, K, MY, MX, 2)).to(dev)
exitw4 = A_fast_full4(S, phase_factors, r, r_min, out2, MY, MX)
for i in range(1):

    start4 = time()
    exitw4 = A_fast_full4(S, phase_factors, r, r_min, out2, MY, MX)
    th.cuda.synchronize(dev)
    end4 = time()

th.backends.cuda.matmul.allow_tf32 = False
exitw3 = A_fast_full5(S, phase_factors2, r[0], r_min, MY, MX)
for i in range(1):
    start7 = time()
    exitw3 = A_fast_full3(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end7 = time()

th.backends.cuda.matmul.allow_tf32 = True
exitw4 = A_fast_full5(S, phase_factors2, r[0], r_min, MY, MX)
for i in range(1):
    start8 = time()
    exitw4 = A_fast_full5(S, phase_factors2, r[0], r_min, MY, MX)
    th.cuda.synchronize(dev)
    end8 = time()

print(f'A_fast_full (custom kernel)                            {end1 - start1}')
print(f'A_fast_full4 (custom kernel 2)                         {end4 - start4}')
print(f'A_fast_full2 (real batched matmul)   NO Tensorfloat32  {end2 - start2}')
print(f'A_fast_full2 (complex batched matmul    Tensorfloat32) {end6 - start6}')
print(f'A_fast_full3 (complex batched matmul NO Tensorfloat32) {end3 - start3}')
print(f'A_fast_full3 (complex batched matmul    Tensorfloat32) {end5 - start5}')
print(f'A_fast_full5 (complex batched matmul NO Tensorfloat32) {end7 - start7}')
print(f'A_fast_full5 (complex batched matmul    Tensorfloat32) {end8 - start8}')
cb = fftshift_checkerboard(MX // 2, MY // 2)

# K = 128
# MY = MX = 96
#split 2
# A_fast_full3 (complex batched matmul NO Tensorfloat32) 1.1578489758539945
# A_fast_full3 (complex batched matmul    Tensorfloat32) 1.1699223290197551
#split
# A_fast_full3 (complex batched matmul NO Tensorfloat32) 0.25746477395296097
# A_fast_full3 (complex batched matmul    Tensorfloat32) 0.25743823405355215

# 256 thread
# A_fast_full5 (complex batched matmul    Tensorfloat32) 0.07920366106554866
# A_fast_full5 (complex batched matmul    Tensorfloat32) 0.07899916195310652
# %%
i = 0
rexitw = th.fft.ifft2(exitw[:, i], norm='ortho')
# rexitw2 = th.ifft(exitw2, 2, True)
#%%

#%%
# plotcxmosaic(fftshift(complex_numpy(rexitw2.cpu()) * cb))
# plotcxmosaic(complex_numpy(exitw2.cpu()))
# #%%
# plotAbsAngle(exitw[0,0].cpu()-exitw2[0].cpu())
# #%%
plotAbsAngle(exitw[0,0].cpu()-exitw3[0].cpu(),'NO tensorfloat')
plotAbsAngle(exitw[0,0].cpu()-exitw4[0].cpu(),'   tensorfloat')
#%%
from smatrix2.util import plotcx
plotcx(exitw3[0].cpu().numpy())