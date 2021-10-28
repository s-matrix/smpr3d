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
from smpr3d.functional import SMatrixSubpix, SparseAmplitudeLoss, SparseSmoothTruncatedAmplitudeLoss, SparseSmoothTruncatedAmplitudeProx
from numpy.fft import fftshift
import torch.optim as optim
from skimage import data

from scipy.ndimage import zoom
from skimage.color import rgb2gray
from skimage.filters import gaussian
A = SMatrixSubpix.apply
smooth_amplitude_loss = SparseSmoothTruncatedAmplitudeLoss.apply
from smatrix.util import *
from scipy.ndimage.interpolation import rotate
# %%
M1 = 60
M = np.array([M1, M1])
s = 256
zoomf = 0.35
amp_range = 0.1
amp_min = 0.9
margins = 0
oa = rgb2gray(data.astronaut())[0:0 + s, 150:150 + s]
oa1 = oa / oa.max()
oa1 = gaussian(oa1, 1 * 1/zoomf)

oa2 = oa1 * amp_range
oa2 += amp_min

oph = rotate(oa1, 0)

ob = zoom(oa2, zoomf) * np.exp(1j * 0.2 * np.pi * zoom(oph, zoomf))

obr = np.zeros((int(ob.shape[0] + 2 * margins), int(ob.shape[1] + 2 * margins))).astype(np.complex64)
obr[margins:margins + ob.shape[0], margins:margins + ob.shape[1]] = ob

obr1 = np.ones((obr.shape[0], obr.shape[1], 2)).astype(np.float32)
obr1[..., 0] = obr.real
obr1[..., 1] = obr.imag
T1 = th.view_as_complex(th.from_numpy(obr1))
N = th.tensor([obr.shape[0], obr.shape[1], 2]).int()

t = T1.numpy()
zplot([t.real, t.imag], cmap=['inferno', 'inferno'], figsize=(9, 5))

r1 = advanced_raster_scan(7, 7, fast_axis=1, mirror=[1, 1],
                         theta=0, dy=5, dx=5)


#%%
E = 300e3
lam = wavelength(E)
defocus_nm = 200
det_pix = 14 * 5e-6
alpha_rad = 3e-3
dx_angstrom = 1.32
q = get_qx_qy_2D_th([M1, M1], [dx_angstrom, dx_angstrom], np.float32, fft_shifted=False).cuda()

from skimage.transform import rescale, downscale_local_mean
fac = 8
ap = fftshift(gaussian(downscale_local_mean(sector_mask(fac*M,fac//2*M,fac//4*M[0]),(fac,fac)),1))
plot(ap)

Ap = th.as_tensor(ap).cuda() * 1e2

Psi_gen = ZernikeProbeSingle(q, lam, fft_shifted=True)

C_target = th.zeros((12)).cuda()
C_target[0] = 5500

C_model = th.zeros((12)).cuda()
C_model[0] = 850

psi_target = Psi_gen(C_target, Ap)
psi_model = Psi_gen(C_target, Ap)

psi_model = th.fft.ifft2(th.view_as_complex(psi_target))
print(psi_model.dtype)
# plotcx(psi_model.cpu().numpy())
plotAbsAngle(psi_model.cpu().numpy())
# %%


qnp = fourier_coordinates_2D([M1,M1], [dx_angstrom,dx_angstrom], centered=False)
q = th.as_tensor(qnp).float().cuda()
Psi_gen = ZernikeProbeSingle(q, lam, fft_shifted=True)
Ap0 = Ap
C1 = C_target
# Psi_model = Psi_gen(C1, Ap0).detach()
# psi_model = th.fft.ifft2(Psi_model, norm='ortho')
#
# Psi_model = Psi_model.unsqueeze(0)
# psi_model = psi_model.unsqueeze(0).cuda()
# psi_model.requires_grad_(False)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(np.angle(psi_model[0].cpu()))
# ax[1].imshow(np.abs(psi_model[0].cpu()))
# plt.show()

# %%

Ap = Ap0.cuda()
q = q.cuda()

# %%
T = T1.unsqueeze(0).cuda()
r = th.from_numpy(r1).cuda()

r[1:] += th.randn_like(r[1:]) * 3
r[r<0] = 0
r[r>30] = 30
psi_model = psi_model.unsqueeze(0)

a_target = A(T, psi_model, r)
I_target = a_target**2
#%%
print(f'psi_model norm: {th.norm(psi_model)**2}')
print(f'I_target norm: {th.sum(I_target[0])}')
#%%
f, ax = plt.subplots()
ax.imshow(a_target[2].cpu())
plt.show()
#%%
plotmosaic(fftshift(a_target.cpu().numpy(),(1,2)),cmap='viridis')
#%%
d2 = Sparse4DData.from_dense(I_target.reshape((7,7,60,60)).cpu().numpy(),make_float=True)
# %%

margin = 0
M = th.tensor(M).int().cuda()
N = th.tensor(th.ceil(r.max(axis=0).values).int())  + M + margin
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


# optimizer = optim.Adam([T_model, C_model, pos], lr=1e-2)
lr = 10e-3
optimizer = optim.SGD([{'params': [S_model], 'lr': lr},
                       {'params': [psi_model], 'lr': lr}], lr=lr, momentum=0.5)

# optimizer = optim.SGD([{'params': [S_model], 'lr': lr}], lr=lr)
# %%



i = 0
it = 1
it = 150


probe_start = 10

S_model.requires_grad = True
psi_model.requires_grad = False
pos.requires_grad = False

n_batches = K // 2
divpoints = array_split_divpoints_ntotal(K, n_batches)
# %%
d = 30
slic = np.s_[:,d:-d,d:-d]
losses = []
errs = []
for i in range(it):
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
        # plotAbsAngle(S_model[0].cpu().detach().numpy(), 'S_model')

        optimizer.step()
        optimizer.zero_grad()

    c = th.vdot(T[slic].ravel(), S_model[slic].ravel())
    T_hat = T * th.exp(-1j*th.angle(c))
    dist = th.norm(S_model[slic] - T_hat[slic])
    x_norm = th.norm(T)
    err = dist / x_norm

    errs.append(err)
    sum_loss /= n_batches
    losses.append(sum_loss)
    print(f'{i:3d}  loss: {sum_loss}    err: {err}')

# print(f'i {i} loss {sum_loss}, C_model = {C_model[0]} , C_target = {C_target[0]}')
# %%
d = margin + M[0] // 2
d = 1
plotAbsAngle(S_model[0, d:-d, d:-d].cpu().detach().numpy(), 'Reconstruction', cmap=['inferno', 'inferno'])
# %%
psi_res = psi_model[0].detach().cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(np.angle(psi_res))
ax[1].imshow(np.abs(psi_res))
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
