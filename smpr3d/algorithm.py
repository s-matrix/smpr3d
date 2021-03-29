# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10a_fasta.ipynb (unless otherwise specified).

__all__ = ['fasta2']

# Cell

import torch as th
from timeit import default_timer as timer
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from .util import *
from .core import SMeta
import numpy as np
from .operators import smatrix_phase_factorsBDK

def fasta2(s_meta: SMeta, A, AH_S, AH_Psi, AH_r, prox_data, Psi_gen: Module,
           a: th.Tensor, S0: th.Tensor, Psi0, C0: th.Tensor, Ap0: th.Tensor, r0: th.Tensor, opts,
           S_sol: th.Tensor = None, Psi_sol: th.Tensor = None, r_sol: th.Tensor = None,
           summary: SummaryWriter = None):
    """
    Block coordinate descent for scanning diffraction experiments with the latent variables
    S: represents the sample, Psi: represents the incoming wave function, r: represents the scanning positions

    :param A:        multi-linear forward operator. must be callable with 3 parameters: A(S, Psi, r)
    :param AH_S:     adjoint of the linear forward operator A_S with all variables fixed but S. Must be callable with 3 parameters: AH_S(z, Psi, r)
    :param AH_Psi:   adjoint of the linear forward operator A_Psi with all variables fixed but Psi. Must be callable with 3 parameters: AH_Psi(z, S, r)
    :param AH_r:     adjoint of the linear forward operator A_r with all variables fixed but r. Must be callable with 3 parameters: AH_r(z, S, Psi)
    :param prox_data:proximal operator for the data term
    :param Psi_gen:      a function that creates the Fourier space probe from aberration coefficients C
    :param a:        float (D, K, MY, MX) measured amplitudes
    :param S0:       float (B_tile, NY, NX, 2) initial S-matrix
    :param C0:       float (12, D) initial aberration coefficients
    :param A0:       float (D, MY, MX) initial apertures
    :param r0:       float (D, K, 2) initial positions
    :param opts:     dictionary of optimization options
    :param S_sol:    optional, float (B_tile, NY, NX, 2) initial S-matrix
    :param Psi_sol:  optional, float (D, MY, MX, 2) initial probes
    :param r_sol:    optional,float (D, K, 2) initial positions
    :param summary:  optional, tensorboard summary writer, used to write intermediate results to tensorboard for vizualisation
    :return: tuple (S, Psi, r), outs, opts
                outs is a dictionary with the keys "R_factors"
                    if a S_sol is provided it has a key "S_errors"
                    if a Psi_sol is provided it has a key "Psi_errors"
    """
    tau_S = opts.tau_S
    tau_Psi = opts.tau_Psi
    tau_r = opts.tau_r

    # lambda operator update speed
    beta = opts.beta
    device = S0.device

    # Maximum number of updates
    max_iters = opts.max_iters
    rank = opts.node_config.rank
    world_size = opts.node_config.world_size
    verbose = opts.verbose

    # Get diffraction pattern shape
    _, _, MY, MX = a.shape

    S_errors = []
    R_factors = []
    Psi_errors = []
    r_errors = []

    # Move probe, probe positions and scattering matrix to the chosen device
    Ap = Ap0.to(device)

    C = C0.to(device)
    C.requires_grad = True

    r = r0.to(device)
    S = S0.to(device)
    if Psi0 is not None:
        Psi = Psi0.to(device)
    else:
        Psi = Psi_gen(C, Ap)

    # use same phase factors for forward and backward
    Psi.phase_factors = smatrix_phase_factorsBDK(Psi, r, s_meta.take_beams, s_meta.q_dft, s_meta.S_shape[0], out=None)

    if rank == 0 and verbose >= 2:
        logging.info(f"a[0,0]**2 sum  : {(a[0, 0] ** 2).sum()}")
        # logging.info(f"z_hat[0,0] norm: {norm(z_hat[0, 0]) ** 2}")

    if world_size == 1:
        ii = 1
        # rexitw = th.ifft(z_hat[:, ii], 2, True)
        # plot_complex_multi(complex_numpy(rexitw.cpu()), f'real space exit waves, position {ii}')

    z1 = th.fft.fft2(A(S, Psi, r), norm='ortho')
    a_model = th.abs(z1)

    # cb = th.as_tensor(fftshift_checkerboard(MY//2,MX//2), device=z1.device)
    # ww = th.fft.fftshift(z1[0,:9],(1,2)) * cb
    #
    # plot_complex_multi(ww.cpu().numpy(), f'real space exit waves, position {ii}')

    # print(a.shape, a_model.shape)
    fac = (1 - (a / (a_model + 1e-3)))
    z1.mul_(fac)
    dz = th.fft.ifft2(z1, norm='ortho')

    S_mom = None
    C_mom = None
    m = [MY // 4, MX // 4]

    S_accel1 = S
    z_accel1 = z1
    alpha1 = 1
    momentum = 0.5
    dampening = 0

    for i in range(max_iters):
        start = timer()

        dS = AH_S(dz, Psi, r)

        if rank == 0 and i == 0:
            logging.info(f'max_abs_dS: {th.max(th.abs(dS))}   max_abs_S: {th.max(th.abs(S.detach()))}')
        if S_mom is None:
            S_mom = th.clone(dS).detach()
        else:
            S_mom.mul_(momentum).add_(dS, alpha=1 - dampening)

        dS.add_(S_mom, alpha=momentum)
        # plotAbsAngle(dS[0, m[0]:-m[0], m[1]:-m[1]].cpu().numpy(), f'dS[{i}]')

        S.add_(dS, alpha=-tau_S)
        del dS
        z1 = th.fft.fft2(A(S, Psi, r), norm='ortho')

        R_fac = R_factor(z1, a, world_size)

        a_model = th.abs(z1)
        fac = (1 - (a / (a_model + 1e-3)))
        z1.mul_(fac)
        dz = th.fft.ifft2(z1, norm='ortho')

        del a_model
        del fac
        del z1

        # solve for probe (step 3 of algorithm 1 from paper)
        if opts.optimize_psi(i):
            # Calculate probe update
            dPsi = AH_Psi(dz, S, r)
            if rank == 0 and i == 0:
                logging.info(f'max_abs_dPsi: {th.max(th.abs(dPsi))}   max_abs_Psi: {th.max(th.abs(Psi.detach()))}')
            # Psi -= tau_Psi * dPsi
            Psi = Psi_gen(C, Ap)
            Psi.backward(dPsi)
            dC = C.grad
            if C_mom is None:
                C_mom = th.clone(C.grad).detach()
            else:
                C_mom.mul_(momentum).add_(C.grad, alpha=1 - dampening)
            # Apply probe update
            dC = dC.add(C_mom, alpha=momentum)
            C = C.detach().clone().add(dC, alpha=- tau_Psi)
            C.requires_grad = True
            if rank == 0:
                logging.info(f'C[0] = {C[0].detach().cpu().numpy()}, C.grad = {dC[0].cpu().numpy()}')
            Psi = Psi_gen(C, Ap)
            Psi.phase_factors = smatrix_phase_factorsBDK(Psi, r, s_meta.take_beams, s_meta.qf, s_meta.q_coords, s_meta.S_shape[0], out=None)

            if verbose == 2 and rank == 0 and summary is not None:
                summary.add_scalar(f'details/max_abs_dPsi', th.max(th.abs(dPsi)), i, timer())
                summary.add_scalar(f'details/max_abs_Psi', th.max(th.abs(Psi.detach())), i, timer())
            del dPsi

        # Logging
        if verbose > 0:
            end = timer()
            R_factors.append(R_fac.cpu().item())
            s = f"{i:03d}/{max_iters:03d} [{(end - start):-02.4f}s] R-factor: {R_factors[-1]:3.3g}   "
            if summary is not None:
                if verbose >= 1 and rank == 0:
                    summary.add_scalar(f'errors/R_factor', R_fac, i, timer())
            if Psi_sol is not None:
                err_Psi = rel_dist(Psi.detach(), Psi_sol)
                Psi_errors.append(err_Psi.cpu().item())
                s += f"err_Psi: {Psi_errors[-1]:3.3g}   "
                if summary is not None and rank == 0:
                    if verbose >= 1:
                        summary.add_scalar(f'errors/probe_error', err_Psi, i, timer())
            if S_sol is not None and rank == 0:
                S[th.isnan(S)] = 0

                err_S = rel_dist(S[:, m[0]:-m[0], m[1]:-m[1]], S_sol[:, m[0]:-m[0], m[1]:-m[1]])
                # plotcxmosaic(complex_numpy(
                #     complex_mul_conj(S[:16, m[0]:-m[0], m[1]:-m[1]], S0[:16, m[0]:-m[0], m[1]:-m[1]]).cpu()),
                #              f'S_sol[{0}]')
                # plotcxmosaic(complex_numpy(S[:16, m[0]:-m[0], m[1]:-m[1]].cpu()), f'S[{i}]')
                # plotAbsAngle(complex_numpy(S[0, m[0]:-m[0], m[1]:-m[1]].cpu() - S_sol[0, m[0]:-m[0], m[1]:-m[1]].cpu()), 'S - S_sol')
                S_errors.append(err_S.cpu().item())
                s += f"err_S: {S_errors[-1]:3.3g}"
                if summary is not None:
                    if verbose >= 1:
                        summary.add_scalar(f'errors/S_error', err_S, i, timer())
            if rank == 0:
                logging.info(s)

    if rank == 0:
        logging.info(f'{i} iterations finished.')

    outs = Param()
    outs.R_factors = np.asarray(R_factors)
    if S_sol is not None:
        outs.S_errors = np.asarray(S_errors)
    if Psi_sol is not None:
        outs.Psi_errors = np.asarray(Psi_errors)
    if r_sol is not None:
        outs.r_errors = np.asarray(r_errors)
    S = S.cpu()
    Psi = Psi.detach().cpu()
    r = r.cpu().numpy()
    C = C.detach().cpu().numpy()

    return (S, Psi, C, r), outs, opts