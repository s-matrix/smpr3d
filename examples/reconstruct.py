# %%
from tqdm import trange
# %matplotlib widget
import matplotlib.pyplot as plt
from smpr3d.core import Sparse4DData, Metadata4D
from pathlib import Path
import numpy as np
# from ipywidgets import AppLayout, FloatSlider, GridspecLayout, VBox, Tab, Box, HBox, IntSlider
# plt.ioff()
import torch as th
from smpr3d.util import *
from smpr3d.functional import SMatrixSubpix, SparseAmplitudeLoss, SparseSmoothTruncatedAmplitudeLoss
from numpy.fft import fftshift
import torch.optim as optim

# %%

# from smpr3d.core import ReconstructionOptions, reconstruct

# metadata2 = Metadata4D(E_ev = E_ev,
#                        alpha_rad = metadata.alpha_rad,
#                        dr=metadata.dr,
#                        k_max = k_max,
#                        rotation_deg = t.rotation_deg)

# options = ReconstructionOptions()

# out = sm.reconstruct(data, metadata, options)

# S = out.smatrix
# r = out.r
# Psi = out.Psi
# R_factor = out.R_factors

# %%

scan_number = 147
base_path = Path('/home/philipp/nvme/insync_berkeley/2020-09-22/')
results_path = base_path / 'results/'
fp = results_path / 'ptycho' / f'{scan_number:3d}_dense.h5'

meta = Metadata4D.from_h5(fp, 'meta')
data = Sparse4DData.from_h5(fp, 'data')
meta, data

# %%

ddense = Sparse4DData.fftshift(data).to_dense(1)

slic_edge = np.s_[:, -50:, :, :]
sds = np.sum(ddense[slic_edge], (0, 1)) / np.prod(data.scan_dimensions)
vacuum_probe = sds * (sds > sds.max() * 20e-2)

# fig, ax = plt.subplots()
# ax.imshow(vacuum_probe)
# # AppLayout(center=fig.canvas)
# plt.show()
# %%
dx = 1 / 2 / meta.k_max

qnp = fourier_coordinates_2D(data.frame_dimensions, dx, centered=False)
q = th.as_tensor(qnp).float().cuda()
Psi_gen = ZernikeProbeSingle(q, meta.wavelength, fft_shifted=True)
Ap0 = th.as_tensor(fftshift(vacuum_probe)).float().cuda()
C1 = th.as_tensor(meta.aberrations).float().cuda()
Psi_model = Psi_gen(C1, Ap0).detach()
psi_model = th.fft.ifft2(Psi_model, norm='ortho')

Psi_model = Psi_model.unsqueeze(0)
psi_model = psi_model.unsqueeze(0).cuda()
psi_model.requires_grad_(False)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(np.angle(psi_model[0].cpu()))
ax[1].imshow(np.abs(psi_model[0].cpu()))
plt.show()

# %%

Ap = Ap0.cuda()
q = q.cuda()

Psi_gen = ZernikeProbeSingle(q, meta.wavelength, fft_shifted=True)

C_target = th.zeros((12)).float().cuda()
C_target[0] = 0

psi_target = Psi_gen(C_target, Ap)

# %%
size = [60, 180]
start = [0, 60]
# size = [60, 60]
# start = [0, 120]
d2 = data.slice(np.s_[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])

f, ax = plt.subplots()
ax.imshow(d2.sum_diffraction())
plt.show()
# %%
r = advanced_raster_scan(d2.scan_dimensions[0], d2.scan_dimensions[1], fast_axis=1, mirror=[1, 1],
                         theta=meta.rotation_deg, dy=meta.pixel_step[0], dx=meta.pixel_step[1])
margin = 10
M = th.tensor(d2.frame_dimensions).int()
N = th.tensor(np.ceil(r.max(axis=0)).astype(np.int32)) + margin + M
K = r.shape[0]

print('N:', N)
print('M:', M)
print('K:', K)

S_model = th.ones((1,) + tuple(N), requires_grad=True, device=th.device('cuda:0'), dtype=th.complex64)

pos = th.as_tensor(r + margin / 2, device=S_model.device)

indices_target = th.as_tensor(d2.indices, device=S_model.device)
counts_target = th.as_tensor(d2.counts, device=S_model.device)
ish = indices_target.shape

indices_target = indices_target.view((K, ish[-1]))
counts_target = th.sqrt(counts_target.view((K, ish[-1])).type(th.float32))

lr = 10e-3
optimizer = optim.SGD([{'params': [S_model], 'lr': lr},
                       {'params': [psi_model], 'lr': lr}], lr=lr)
# optimizer = optim.Adam([T_model, C_model, pos], lr=1e-2)
# optimizer = optim.SGD([{'params': [S_model], 'lr': 1e-4},
#                        {'params': [psi_model], 'lr': 1e-4}], lr=1e-4)
# optimizer = optim.SGD([{'params': [S_model], 'lr': 1e-3}], lr=1e-3)
# %%

A = SMatrixSubpix.apply
smooth_amplitude_loss = SparseSmoothTruncatedAmplitudeLoss.apply

i = 0
it = 1
it = 10

probe_start = 2

S_model.requires_grad = True
psi_model.requires_grad = False
pos.requires_grad = False

n_batches = K // 16
divpoints = array_split_divpoints_ntotal(K, n_batches)
# %%
for i in trange(it):
    sum_loss = 0
    random_order = th.randperm(K)
    # grad_Psi_model = th.zeros_like(Psi_model, requires_grad=False)
    if i > probe_start:
        psi_model.requires_grad = True

    for b in range(n_batches):
        take_ind = random_order[divpoints[b]:divpoints[b + 1]]

        a_model = A(S_model, psi_model, pos[take_ind])

        loss = smooth_amplitude_loss(a_model, indices_target[take_ind], counts_target[take_ind])
        loss_sum = loss.mean()
        sum_loss += loss_sum.item()
        loss_sum.backward()

        # if i > probe_start:
        #     plotAbsAngle(psi_model.grad[0].cpu().detach().numpy(),'psi_model.grad')

        optimizer.step()
        optimizer.zero_grad()

    d = margin + M[0] // 2
    plotAbsAngle(S_model[0, d:-d, d:-d].cpu().detach().numpy(),f'it {i:3d}')
    sum_loss /= n_batches
    print(f'loss: {sum_loss}')

# print(f'i {i} loss {sum_loss}, C_model = {C_model[0]} , C_target = {C_target[0]}')
# %%
d = margin + M[0] // 2
plotAbsAngle(S_model[0, d:-d, d:-d].cpu().detach().numpy())
# %%
psi_res = psi_model[0].detach().cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(np.angle(psi_res))
ax[1].imshow(np.abs(psi_res))
plt.show()
# %%
psi_res = fftshift(th.fft.fft2(psi_model[0], norm='ortho').detach().cpu().numpy() * fftshift_checkerboard(M[0]//2,M[0]//2))
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(imsave(psi_res))
plt.show()
# %%
# from skimage.transform import rotate
#
# m = 80
# t = complex_numpy(T_model.clone().detach().cpu())
# t = rotate(t.real, -theta) + 1j * rotate(t.imag, -theta)
# # plotcx(t[m:-m,m:-m])
# zplot([np.abs(t)[m:-m, m:-m], np.angle(t)[m:-m, m:-m]], title=['Abs', 'Angle'], cmap=['inferno', 'inferno'],
#       figsize=(9, 5))
#
# %%

it

# %%
