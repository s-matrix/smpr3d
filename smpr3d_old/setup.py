# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20_setup.ipynb (unless otherwise specified).

__all__ = ['setup_logging', 'configure_node', 'load_smatrix_data_list2', 'initial_probe_amplitude',
           'prepare_beam_parameters', 'initial_smatrix', 'report_smatrix_parameters', 'report_initial_probes']

# Cell
import logging
import sys

def setup_logging(path, log_filename):
    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler("{0}/{1}.log".format(path, log_filename))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

# Cell

import os
from .util import *
import torch as th
import torch.distributed as dist
from numba import cuda
import GPUtil
import psutil
import numpy as np
import h5py as h5

def configure_node(dist_backend, init_method):
    args = Param()
    args.dist_backend = dist_backend

    is_using_slurm = os.environ.get('SLURM_NTASKS') is not None
    if is_using_slurm:
        SLURM_LOCALID = int(os.environ.get('SLURM_LOCALID'))
        SLURM_PROCID = int(os.environ.get('SLURM_PROCID'))
        SLURM_NTASKS = int(os.environ.get('SLURM_NTASKS'))
        SLURM_NTASKS_PER_NODE = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
        # logging.info(f'SLURM_LOCALID: {SLURM_LOCALID}')
        # logging.info(f'SLURM_PROCID: {SLURM_PROCID}')
        # logging.info(f'SLURM_NTASKS: {SLURM_NTASKS}')
        # logging.info(f'SLURM_NTASKS_PER_NODE: {SLURM_NTASKS_PER_NODE}')
        args.slurm_tasks_per_node = SLURM_NTASKS_PER_NODE if SLURM_NTASKS_PER_NODE is not None else 0
        args.rank = SLURM_PROCID if SLURM_PROCID is not None else 0
        args.gpu = SLURM_LOCALID if SLURM_LOCALID is not None else args.rank
        args.world_size = SLURM_NTASKS if SLURM_NTASKS is not None else 1
        args.is_distributed = True
        args.scheduler = 'slurm'
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.is_distributed = False
        args.scheduler = 'local'

    args.device = th.device(f'cuda:{args.gpu}')
    # if args.world_size > 1:
        # let numba know which device we are running the kernels on
    logging.info(f'rank {args.rank} avail gpus: {[x.id for x in GPUtil.getGPUs()]}')
    logging.info(f'Selecting device: {args.gpu}')
    cuda.select_device(args.gpu)
    # dist.init_process_group(backend=args.dist_backend, rank=args.rank, world_size=args.world_size,
    #                         init_method=init_method)

    ram_gpu_free_GB = []
    ram_cpu_free_GB = psutil.virtual_memory().available / 2 ** 30
    gpus = GPUtil.getGPUs()
    gpu = gpus[args.gpu]
    ram_gpu_free_GB = gpu.memoryFree / 1000
    if args.rank == 0:
        logging.info(f'Scheduler: {args.scheduler}')
        logging.info(f'System resources:')
    logging.info(f'Rank {args.rank}    Free CPU RAM: {ram_cpu_free_GB} GB')
    # if args.world_size > 1:
    #     dist.barrier()
    logging.info(f'Rank {args.rank}    Free GPU RAM: {ram_gpu_free_GB} GB')
    # if args.world_size > 1:
    #     dist.barrier()
    logging.info(f'Rank {args.rank} is using device {args.gpu}/{len(gpus)}: {gpu.name} driver: v{gpu.driver}')
    # if args.world_size > 1:
    #     dist.barrier()
    return args

# Cell
from numpy.fft import fftshift

def load_smatrix_data_list2(fn, device, rank, world_size, fftshift_data=False, subset=None, r_max=None):
    """
    Load S-matrix reconstruction data. Required hdf5 keys:

    Abbreviations:
    D: number of scans/ aperture functions
    K: number of scan positions
    MY/MX: detector shape
    NY/NX: S-matrix shape

    lambda:                         electron wavelength in Angstrom                 scalar float
    alpha_rad:                      convergence angle in radians                    scalar float
    C:                              aberration coefficients                         (12, D) float
    k_max:                          real-space Nyquist resolution (half-period)     (2,) float
    specimen_thickness_angstrom:    appoximate thickness in Angstrom                scalar float
    vacuum_probe:                   an image of the vacuum beam                     (MY, MX) float
    data:                           diffraction data, fft_shifted                   (D, K, MY, MX) float or integer
    r:                              real-space positions, in pixel coordinates      (D, K, 2) float
    probe_fourier:                  initial aperture functions                      (D, MY, MX) complex

    Optional hdf5 keys:

    relative_shifts:                relative shifts of the positions for different aberrations  (D, 2) float
    S_target                        target S-matrix from simulation                             (B, NY, NX) complex
    Psi_target                      target aperture functions from simulation                   (D, MY, MX) complex
    r_target                        target positions from simulation, in pixel coordinates      (D, K, 2) float

    :param fn: file name. Must be a valid hdf5 file
    :param device: target device for all data
    :param rank: rank of the loading process
    :param world_size: world_size of the current program
    :return: all needed data
    """

    if rank == 0:
        logging.info(f'Now      loading data file {fn}')
    with h5.File(fn, 'r') as f:
        D = len(subset)
        lam = f['lambda'][0]
        # step = f['skip'][()]
        alpha_rad = f['alpha_rad'][0]

        if subset is None:
            C = th.from_numpy(f['C'][:, :]).to(device)
            subset = np.arange(D)
        else:
            C = th.from_numpy(f['C'][:, subset]).to(device)

        dx = th.from_numpy(f['d_sm'][:])
        specimen_thickness_angstrom = f['specimen_thickness_angstrom'][0]
        vac = f['vacuum_probe'][:, :]

        if fftshift_data:
            vac = fftshift(vac)
        vacuum_probe = th.from_numpy(vac).to(device)

        data = []
        for d in subset:
            data.append(f[f'data_{d}'][:, :, :])

        if fftshift_data:
            for d in range(D):
                data[d] = fftshift(data[d], (1, 2))

        data = np.array(data)

        r0 = []
        for d in subset:
            r0.append(f[f'r_{d}'][:, :])

        r0 = np.array(r0)

        try:
            r_rel = f['relative_shifts'][...]
            if rank == 0:
                logging.info(f'Relative shifts of datasets:')
                for rr in r_rel:
                    logging.info(f'     {rr}')
            for d in range(D):
                r0[d] += r_rel[d, :]
        except:
            pass

        y_min = 1e6
        x_min = 1e6
        for d in range(D):
            y_min1 = np.min(r0[d][:, 0])
            x_min1 = np.min(r0[d][:, 1])
            y_min = np.min([y_min, y_min1])
            x_min = np.min([x_min, x_min1])

        for d in range(D):
            r0[d] -= [y_min, x_min]

        K = data.shape[1]
        MY = data.shape[2]
        MX = data.shape[3]
        detector_shape = np.array([MY, MX])

        divpts = array_split_divpoints(data, world_size, 1)
        d = data[:, divpts[rank]:divpts[rank + 1], :, :]
        I_target = th.from_numpy(d).to(device)
        r = th.from_numpy(r0[:, divpts[rank]:divpts[rank + 1], :]).to(device)

        y_max = -1e6
        x_max = -1e6
        for d in range(D):
            y_max1 = np.max(r0[d][:, 0])
            x_max1 = np.max(r0[d][:, 1])
            y_max = np.max([y_max, y_max1])
            x_max = np.max([x_max, x_max1])
        y_min = 1e6
        x_min = 1e6
        for d in range(D):
            y_min1 = np.min(r0[d][:, 0])
            x_min1 = np.min(r0[d][:, 1])
            y_min = np.min([y_min, y_min1])
            x_min = np.min([x_min, x_min1])

        if rank == 0:
            logging.info(f'Position array boundaries: [{y_min}:{y_max},{x_min}:{x_max}]')
            if y_min < 0 or x_min < 0:
                logging.warning(f'y_min = {y_min}, x_min = {x_min}, NEGATIVE INDICES ARE NOT ALLOWED!')
        if rank == 0:
            logging.info(f'memory allocated: {th.cuda.memory_allocated()/1024**2} MB')

        K_rank = I_target.shape[1]

        S_sol = None
        Psi_sol = None
        r_sol = None
        try:
            S_sol = th.as_tensor(f['S_target'][:, :, :], dtype=th.complex64).to(device)
        except:
            S_sol = None

        try:
            Psi_sol = []
            for d in range(D):
                Psi_sol.append(th.as_tensor(f[f'Psi_target_{d}'][:, :], dtype=th.complex64).to(device))
        except:
            Psi_sol = None

        Psi_sol = th.stack(Psi_sol,0)

        try:
            r_sol = []
            for d in range(D):
                r_sol.append(th.from_numpy(f[f'r_target_{d}'][:, :]).to(device))
        except:
            r_sol = None

        r_sol = th.stack(r_sol,0)
        # Psi = []
        # for d in range(D):
        #     Psi.append(f[f'Psi0_{d}'][:, :])
        #
        # if fftshift_data:
        #     for d in range(D):
        #         Psi[d] = fftshift(Psi[d], (1, 2))
        #         cb = fftshift_checkerboard(MY // 2, MX // 2)
        #         Psi[d] *= cb
        #
        # Psi0 = []
        # for d in subset:
        #     Psi0.append(cx_from_numpy(Psi[d]).to(device).squeeze())

        # S-matrix lateral dimensions.
        # Ensure that the scattering matrix real space sampling is identical to
        # that implied by the maximum reciprocal lattice vector of the
        # diffraction pattern.
        MY_max = np.max(np.array(MY))
        MX_max = np.max(np.array(MX))

        NY = int((np.ceil((y_max + MY_max) / MY_max) * MY_max).item())
        NX = int((np.ceil((x_max + MX_max) / MX_max) * MX_max).item())
        fy = NY // MY_max
        fx = NX // MX_max
        if rank == 0:
            logging.info(f'Finished loading data file {fn}')

        # Enforce same data for all arrays
        da = th.float32
        vacuum_probe = vacuum_probe.type(da)

        r = r.type(da)
        C = C.type(da)

        # for d in subset:
        #     Psi0[d] = Psi0[d].type(da)

        if r_sol is not None:
            r_sol = r_sol.type(da)
        return lam, alpha_rad, C, dx, specimen_thickness_angstrom, vacuum_probe, D, K, K_rank, MY, MX, NY, NX, \
               fy, fx, detector_shape, r, I_target, y_max, x_max, y_min, x_min, S_sol, Psi_sol, r_sol

# Cell
def initial_probe_amplitude(vacuum_probe, I_target, world_size, rank):
    D, K, MY, MX = I_target.shape
    # D x My x Mx
    I_mean = th.sum(I_target, 1)
    if world_size > 1:
        dist.all_reduce(I_mean, op=dist.ReduceOp.SUM)
    I_mean /= K

    if rank == 0:
        logging.info(f'I_mean             :{th.sum(I_mean, (1, 2)).cpu().numpy()}')

    # dim: D x K_rank
    # total intensity per diffraction pattern
    I_tot = th.sum(I_target, (2, 3))

    # max intensity over all diffraction patterns
    I_max, I_max_inds = th.max(I_tot, 1)
    if world_size > 1:
        dist.all_reduce(I_max, op=dist.ReduceOp.MAX)

    # max intensity over all diffraction patterns, for each defocus
    # dim: D
    if rank == 0:
        logging.info(f'I_max              :{I_max.cpu().numpy()}')

    # dim: D
    I_init = vacuum_probe.unsqueeze(0).repeat((D, 1, 1))
    I_norm = I_init.norm(1, dim=(1, 2))
    fac = I_max / I_norm
    I_init *= fac[:, None, None]
    if rank == 0:
        logging.info(f'I_init norm        :{I_init.norm(1, dim=(1, 2)).cpu().numpy()}')
    A_init = th.sqrt(I_init)

# Cell
def prepare_beam_parameters(take_beams, q2, specimen_thickness_angstrom, alpha_rad, lam, max_phase_error,
                            use_full_smatrix, device):
    # number of beams
    B = th.sum(take_beams).item()
    beam_numbers = th.ones_like(take_beams, dtype=th.long, device=device) * -1
    beam_numbers[take_beams] = th.arange(0, B, device=device)

    if use_full_smatrix:
        # override tiling of the aperture
        reduction_factor = 1.0
        B_tile = B
        tile_order = beam_numbers
    else:
        raise NotImplementedError('coming soon')

    if reduction_factor == 1:
        tile_map = beam_numbers[take_beams]
        tile_number = beam_numbers
    else:
        tile_map = -1
        tile_number = -1

    return B, B_tile, tile_order, tile_number, tile_map

# Cell
from numpy.fft import fftfreq

def initial_smatrix(shape, q_space_tiling, device, is_unitary, include_plane_waves, initial_depth=0, lam=0, q2=0,
                    dtype=th.complex64, is_pinned=False):
    """

    :param shape:               (4-tuple) shape of the S-matrix
    :param q_space_tiling:      (2D)
    :param device:              torch.device
    :param is_unitary:          bool, if the S-matrix should be unitary
    :param include_plane_waves: bool, if the S-matrix should be in the plane-wave basis
    :param initial_depth:       float, z-depth of the initial S-matrix in Angstrom, gives quadratic phase offset to beams
    :param lam:                 float, wavelength in Angstrom
    :param q2:                  2D, wavevector squared
    :param dtype:               th.dtype
    :param is_pinned:           bool, create pinned memory for CPU
    :return: initialized S-matrix
    """
    B, NY, NX = shape
    MY, MX = q_space_tiling.shape
    fy, fx = NY // MY, NX // MX
    S = th.zeros(shape, dtype=dtype, device=device)
    if is_pinned:
        S = S.pin_memory()
    if initial_depth > 0:
        take_beams = q_space_tiling >= 0
        tile_map = q_space_tiling[take_beams]
        depth_init = th.exp(1j* -np.pi * q2 * lam * initial_depth).to(device)
        for b in range(B):
            S[b] = th.mean(depth_init[take_beams][tile_map == b], axis=0)
    else:
        depth_init = th.zeros(q2.shape)
    if include_plane_waves:
        qx, qy = np.meshgrid(fftfreq(MX), fftfreq(MY))
        q = np.array([qy, qx])
        q_dft = th.from_numpy(q).to(device).type(S.dtype)
        coords = th.from_numpy(fftshift(np.array(np.mgrid[-MY // 2:MY // 2, -MX // 2:MX // 2]), (1, 2))).to(device)
        for b in range(B):
            cur_beam = q_space_tiling == b
            cur_beam = cur_beam[None, ...].expand_as(coords)
            c = coords[cur_beam]
            cur_planewave = th.exp(2j * np.pi * (q_dft[0] * c[0] + q_dft[1] * c[1]))
            S[b] = S[b] * cur_planewave.repeat((fy, fx))
    # S += th.randn_like(S) * 1e-10
    if is_unitary:
        sy = NY // 2
        sx = NX // 2
        CS = S[:, sy:sy + MY, sx:sx + MX]
        css = th.norm(CS,p=2, dim=(1, 2))
        S /= css[:, None, None]
    return S, depth_init

# Cell
from .util import memory_mb
def report_smatrix_parameters(rank, world_size, a, S, B, D, K, MY, MX, NY, NX, fy, fx, B_tile, K_rank,
                              specimen_thickness_angstrom, depth_init, y_max, x_max, y_min, x_min):
    if rank == 0:
        nonzero_measurements = float(th.sum(a.cpu() > 0))
        y_size = (y_max - y_min)
        x_size = (x_max - x_min)
        unknowns = y_size * x_size * B_tile
        logging.info(f"B (number of beams)      = {B}")
        logging.info(f'B_tile (number of tiles) = {B_tile}')
        logging.info(f"K (number of positions)  = {K}")
        logging.info(f"D (number of aberrations)= {D}")
        logging.info(f"M (detector size)        = {MY, MX}")
        logging.info(f"N (S-matrix size)        = {NY, NX}")
        logging.info(f"f (PRISM factor)         = {fy, fx}")
        logging.info(f"")
        logging.info(f'tile reduction_factor    = {B_tile / B}')
        logging.info(f'#unknowns                = {unknowns:3.3g}')
        logging.info(f'#nonzero_measurements    = {nonzero_measurements:3.3g}')
        logging.info(f'variable oversampling    = {nonzero_measurements / unknowns:-2.2f}')
        logging.info(f"")
        logging.info(f"initial thickness est.   = {specimen_thickness_angstrom / 10} nm")

        size_smatrix_gb = memory_mb((B_tile, NY, NX, 2), th.float32)
        size_exitwaves_gb = memory_mb((K_rank, D, MY, MX, 2), th.float32)
        logging.info(f"")
        logging.info(f"Memory needed for S-matrix    : {size_smatrix_gb:-2.1f} MB")
        logging.info(f"Memory needed for measurements: {size_exitwaves_gb / 2:-2.1f} MB")
        logging.info(f"Memory needed for exitwaves   : {size_exitwaves_gb:-2.1f} MB")
        logging.info(f"")
        if world_size > 1:
            dist.barrier()
        logging.info(f"rank {rank:-3d}  K_rank  = {K_rank:-4d}")
    else:
        if world_size > 1:
            dist.barrier()
        logging.info(f"rank {rank:-3d} K_rank  = {K_rank:-4d}")

# Cell
def report_initial_probes(rank, world_size, Psi_gen, C, A_init, specimen_thickness_angstrom, q, lam,
                          alpha_rad, summary=None):
    C_exit = C.clone()
    C_exit[0] += specimen_thickness_angstrom

    Psi_model = Psi_gen(C, A_init).detach()
    psi_model = th.ifft(Psi_model, 2, True)
    D, MY, MX, _ = Psi_model.shape
    cb = fftshift_checkerboard(MY // 2, MX // 2)

    if rank == 0:
        logging.info(f"Psi_model norm: {th.norm(Psi_model[0]) ** 2}")

    Psi_exit = Psi_gen(C_exit, A_init).detach()
    psi_exit = th.ifft(Psi_exit, 2, True)

    if world_size == 1:
        fig_Psi = plot_complex_multi(cb * fftshift(Psi_model.cpu(), (1, 2)),
                                     'Fourier space entrance surface probes')
        plot_complex_multi(psi_model.cpu(), 'Real Space entrance surface probes')
        plot_complex_multi(cb * fftshift(Psi_exit.cpu(), (1, 2)), f'Fourier space exit surface probes')
        plot_complex_multi(psi_exit.cpu(), f'Real Space exit surface probes')

        if summary is not None:
            summary.add_figure('init/Psi_entrance', fig_Psi, 0, close=True)
    else:
        if rank == 0:
            fig_Psi = plot_complex_multi(cb * fftshift(Psi_model.cpu(), (1, 2)),
                                         'Fourier Space entrance surface probes', show=False)
            fig_psi = plot_complex_multi(psi_model.cpu(), 'Real Space entrance surface probes', show=False)
            fig_Psi_ex = plot_complex_multi(cb * fftshift(Psi_exit.cpu(), (1, 2)),
                                            f'Fourier    exit surface probes', show=False)
            fig_psi_ex = plot_complex_multi(psi_exit.cpu(), f'Real Space exit surface probes',
                                            show=False)
            if summary is not None:
                summary.add_figure('init/Psi_entrance', fig_Psi, 0, close=True)
                summary.add_figure('init/psi_entrance', fig_psi, 0, close=True)
                summary.add_figure('init/Psi_exit', fig_Psi_ex, 0, close=True)
                summary.add_figure('init/psi_exit', fig_psi_ex, 0, close=True)