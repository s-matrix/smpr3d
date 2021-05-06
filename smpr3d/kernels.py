# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/40_operators.kernels.ipynb (unless otherwise specified).

__all__ = ['smatrix_forward_kernel', 'smatrix_backward_kernel_S', 'phase_factor_kernelDBK', 'phase_factor_kernelKB',
           'smatrix_forward_kernel_fast_full4', 'split_kernel', 'split_kernel4', 'split_kernel5', 'split_kernel2',
           'split_kernel3', 'overlap_kernel_real2', 'psi_denom_kernel', 'psi_kernel', 'A_realspace_kernel',
           'AtF2_kernel']

# Cell
import numba.cuda as cuda
import cmath as cm

@cuda.jit
def smatrix_forward_kernel(S, phase_factors, rho, r_min, out):
    """
    :param S:               B x NY x NX
    :param phase_factors:   B x D x K x 2
    :param rho:               D x K x 2
    :param out:             D x K x MY x MX
    :return: exit waves in out
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    D, K, MY, MX, _ = out.shape
    B = S.shape[0]
    MM = MY * MX

    b = int(n // MM)
    my = (n - b * MM) // MX
    mx = (n - b * MM - my * MX)

    if n < B * MY * MX:
        for d in range(D):
            for k in range(K):
                # indexing with pixel precision
                rho0 = int(rho[d, k, 0] - r_min[0])
                rho1 = int(rho[d, k, 1] - r_min[1])

                a = S[b, rho0 + my, rho1 + mx, 0]
                c = S[b, rho0 + my, rho1 + mx, 1]
                u = phase_factors[b, d, k, 0]
                v = phase_factors[b, d, k, 1]

                val_real = a * u - c * v
                val_imag = c * u + a * v

                cuda.atomic.add(out, (d, k, my, mx, 0), val_real)
                cuda.atomic.add(out, (d, k, my, mx, 1), val_imag)

# Cell
@cuda.jit
def smatrix_backward_kernel_S(z, phase_factors, mean_probe_intensities, r, r_min, out, tau):
    """
    S-matrix has beam tilts included, pre-calculated scanning phase factors.
    Fastest to compute

    :param z:                       D x K x MY x MX x 2
    :param phase_factors:           B x D x K x 2
    :param r:                       D x K x 2
    :param mean_probe_intensities:  D
    :param out:                     B x NY x NX x 2
    :param z_strides:  (4,)
    :return: exit waves in out
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    D, K, MY, MX, _ = z.shape
    B = out.shape[0]

    b = n // (MY * MX)
    my = (n - b * (MX * MY)) // MX
    mx = (n - b * (MX * MY) - my * MX)

    if n < MY * MX * B:
        for d in range(D):
            for k in range(K):
                r0 = int(r[d, k, 0] - r_min[0])
                r1 = int(r[d, k, 1] - r_min[1])

                a = z[d, k, my, mx, 0]
                c = z[d, k, my, mx, 1]
                u = phase_factors[b, d, k, 0]
                v = phase_factors[b, d, k, 1]

                val_real = a * u + c * v
                val_imag = c * u - a * v

                val_real *= tau[0] / mean_probe_intensities[d]
                val_imag *= tau[0] / mean_probe_intensities[d]

                cuda.atomic.add(out, (b, r0 + my, r1 + mx, 0), val_real)
                cuda.atomic.add(out, (b, r0 + my, r1 + mx, 1), val_imag)

# Cell
@cuda.jit
def phase_factor_kernelDBK(Psi, rho, qB, out):
    """
    Calculate the phase factors (due to beam scan) probe wave function so that
    the probe is scanned to the correct place for each diffraction pattern

    :param Psi:         D x B
        Probe wave functions Fourier coefficient for each beam to be mutliplied
        by phase factor to account for beam scan position
    :param rho:           D x K x 2
        Probe positions in pixels
    :param qB:          2 x B
        Fourier space coordinates of the beams
    :param out:         D x B x K x 2
        Phase factors output
    :param out_strides: (3,)
    :return: scanning phases for all defoc, beams, positions
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B, D, K, c = out.shape
    N = B * K * D

    b = n // (K * D)
    d = (n - b * (K * D)) // K
    k = (n - b * (K * D) - d * K)

    if n < N:
        rho0 = rho[d, k, 0]
        rho1 = rho[d, k, 1]
        Psic = Psi[d, b, 0] + 1j * Psi[d, b, 1]
        # scanning phase with subpixel precision
        v = cm.exp(-2j * cm.pi * (qB[0, b] * rho0 + qB[1, b] * rho1)) * Psic
        out[b, d, k, 0] = v.real
        out[b, d, k, 1] = v.imag

# Cell
@cuda.jit
def phase_factor_kernelKB(Psi, rho, qB, out):
    """
    Calculate the phase factors (due to beam scan) probe wave function so that
    the probe is scanned to the correct place for each diffraction pattern

    :param Psi:         B x 2
        Probe wave functions Fourier coefficient for each beam to be mutliplied
        by phase factor to account for beam scan position
    :param rho:         K x 2
        Probe positions in pixels
    :param qB:          2 x B
        Fourier space coordinates of the beams
    :param out:         K x B x 2
        Phase factors output
    :return: scanning phases for all defoc, beams, positions
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K, B, c = out.shape
    N = B * K

    b = n // (K)
    k = (n - b * K)

    if n < N:
        rho0 = rho[k, 0]
        rho1 = rho[k, 1]
        Psic = Psi[b, 0] + 1j * Psi[b, 1]
        # scanning phase with subpixel precision
        v = cm.exp(-2j * cm.pi * (qB[0, b] * rho0 + qB[1, b] * rho1)) * Psic
        out[k, b, 0] = v.real
        out[k, b, 1] = v.imag

# Cell
@cuda.jit
def smatrix_forward_kernel_fast_full4(S, phase_factors, r, r_min, out):
    """

    :param S:               B x NY x NX

    :param phase_factors:   B x D x K x 2
    :param r:               D x K x 2
    :param out:             D x K x MY x MX
    :param out_strides: (4,)
    :return: exit waves in out
    """
    k, my, mx = cuda.grid(3)
    D, K, MY, MX, _ = out.shape
    B = S.shape[0]

    if k < K and my < MY and mx < MX:
        for d in range(D):
            accum_real = 0.
            accum_imag = 0.
            for b in range(B):
                rho0 = int(r[d, k, 0] - r_min[0])
                rho1 = int(r[d, k, 1] - r_min[1])
                S_b_real = S[b, rho0 + my, rho1 + mx, 0]
                S_b_imag = S[b, rho0 + my, rho1 + mx, 1]
                a = S_b_real
                c = S_b_imag
                u = phase_factors[b, d, k, 0]
                v = phase_factors[b, d, k, 1]

                accum_real += a * u - c * v
                accum_imag += c * u + a * v

            out[d, k, my, mx, 0] = accum_real
            out[d, k, my, mx, 1] = accum_imag

@cuda.jit
def split_kernel(S, r, out):
    """

    :param S: B x NY x NX x 2
    :param r: K x2
    :param out: K x MY x MX x 2
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K, MY, MX, B, _ = out.shape
    N = K * MY * MX * B

    k = n // (MY * MX * B)
    my = (n - k * MY * MX * B) // (MX * B)
    mx = (n - k * MY * MX * B - my * MX * B) // B
    b = (n - k * MY * MX * B - my * MX * B - mx * B)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        out[k, my, mx, b, 0] = S[b, y + my, x + mx, 0]
        out[k, my, mx, b, 1] = S[b, y + my, x + mx, 1]

@cuda.jit
def split_kernel4(S, r, out):
    """

    :param S:   B x NY x NX
    :param r:   K x 2
    :param out: B x K x MY x MX
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B, K, MY, MX = out.shape
    N = K * MY * MX * B

    b = n // (MY * MX * K)
    k = (n - b * MY * MX * K) // (MX * MY)
    my = (n - b * MY * MX * K - k * MX * MY) // MX
    mx = (n - b * MY * MX * K - k * MX * MY - my * MX)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        out[b, k, my, mx] = S[b, y + my, x + mx]

@cuda.jit
def split_kernel5(S, r, out):
    """

    :param S: B x NY x NX x 2
    :param r: K x 2
    :param out: K x B x MY x MX x 2
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K, B, MY, MX, _ = out.shape
    N = K * MY * MX * B

    k = n // (MY * MX * B)
    b = (n - k * MY * MX * B) // (MX * MY)
    my = (n - k * MY * MX * B - k * MX * MY) // MX
    mx = (n - k * MY * MX * B - k * MX * MY - my * MX)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        out[k, b, my, mx, 0] = S[b, y + my, x + mx, 0]
        out[k, b, my, mx, 1] = S[b, y + my, x + mx, 1]

@cuda.jit
def split_kernel2(S, r, out):
    """

    :param S: B x NY x NX x 2
    :param r: K x2
    :param out: K x MY x MX x 2
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K, MY, MX, B, _ = out.shape
    N = K * MY * MX * B

    k = n // (B)
    b = (n - k * B)

    if n < N:
        for my in range(MY):
            for mx in range(MX):
                y = r[k, 0]
                x = r[k, 1]
                out[k, my, mx, b, 0] = S[b, y + my, x + mx, 0]
                out[k, my, mx, b, 1] = S[b, y + my, x + mx, 1]

@cuda.jit
def split_kernel3(S, r, out):
    """
    :param S: B x NY x NX x 2
    :param r: K x2
    :param out: K x MY x MX x 2
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K, MY, MX, B, _ = out.shape
    N = K * MY * MX * B

    k = n // (MY * B)
    my = (n - k * MY * B) // (B)
    b = (n - k * MY * B - my * B)

    if n < N:
            for mx in range(MX):
                y = r[k, 0]
                x = r[k, 1]
                out[k, my, mx, b, 0] = S[b, y + my, x + mx, 0]
                out[k, my, mx, b, 1] = S[b, y + my, x + mx, 1]

# Cell
@cuda.jit
def overlap_kernel_real2(r, z, out):
    """

    :param r: K x 2
    :param z: BB x K x MY x MX
    :param out: BB x NY x NX
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K = r.shape[0]
    BB, MY, MX = z.shape
    N = BB * K * MY * MX

    bb = n // (K * MY * MX)
    k = (n - bb * (K * MY * MX)) // (MY * MX)
    my = (n - bb * (K * MY * MX) - k * MY * MX) // MX
    mx = (n - bb * (K * MY * MX) - k * MY * MX - my * MX)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        val = z[bb, my, mx]
        cuda.atomic.add(out, (bb, y + my, x + mx), val)

# Cell
@cuda.jit
def psi_denom_kernel(r, t, out):
    """

    :param r: K x 2
    :param t: BB x NY x NX
    :param out: BB x MY x MX
    :return:
    """

    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K = r.shape[0]
    BB, MY, MX = out.shape
    N = BB * K * MY * MX

    bb = n // (K * MY * MX)
    k = (n - bb * (K * MY * MX)) // (MY * MX)
    my = (n - bb * (K * MY * MX) - k * (MY * MX)) // MX
    mx = (n - bb * (K * MY * MX) - k * (MY * MX) - my * MX)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        val = abs(t[bb, y + my, x + mx]) ** 2
        cuda.atomic.add(out, (bb, my, mx), val)

# Cell
@cuda.jit
def psi_kernel(r, t, z, out):
    """

    :param r:   K x 2
    :param t:   BB x NY x NX
    :param z:   K x MY x MX
    :param out: BB x MY x MX
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    K = r.shape[0]
    MY, MX = out.shape
    N = K * MY * MX

    k = (n // (MY * MX))
    my = ((n - k * (MY * MX)) // MX)
    mx = ((n - k * (MY * MX) - my * MX))

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        t_conj = t[y + my, x + mx].conjugate()
        val = t_conj * z[k, my, mx]
        cuda.atomic.add(out.real, (my, mx), val.real)
        cuda.atomic.add(out.imag, (my, mx), val.imag)

# Cell
@cuda.jit
def A_realspace_kernel(r, t, psi, out):
    """

    :param r:   K x 2
    :param t:   B x NY x NX
    :param psi: B x K x MY x MX
    :param out: K x MY x MX
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B, K, MY, MX, _ = psi.shape
    N = K * MY * MX

    k = n // (MY * MX)
    my = (n - k * (MY * MX)) // MX
    mx = (n - k * (MY * MX) - my * MX)

    if n < N:
        for bb in range(B):
            y = r[k, 0]
            x = r[k, 1]

            # val = t[bb, y + my, x + mx] * psi[bb, k, my, mx]
            # cuda.atomic.add(out.real, (k, y + my, x + mx), val.real)
            # cuda.atomic.add(out.imag, (k, y + my, x + mx), val.imag)
            #
            a = t[bb, y + my, x + mx, 0]
            b = t[bb, y + my, x + mx, 1]
            u = psi[bb, k, my, mx, 0]
            v = psi[bb, k, my, mx, 1]

            val_real = a * u - b * v
            val_imag = b * u + a * v

            cuda.atomic.add(out, (k, my, mx, 0), val_real)
            cuda.atomic.add(out, (k, my, mx, 1), val_imag)

# Cell
@cuda.jit
def AtF2_kernel(z, psi, r, out):
    """

    :param z:   K x MY x MX
    :param psi: B x K x MY x MX
    :param r:   K x 2
    :param out: B x NY x NX
    :return:
    """

    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B, K, MY, MX = psi.shape
    N = B * K * MY * MX

    bb = (n // (MY * MX * K))
    k = (n - bb * (MY * MX * K)) // (MY * MX)
    my = (n - bb * (MY * MX * K) - k * (MY * MX)) // MX
    mx = (n - bb * (MY * MX * K) - k * (MY * MX) - my * MX)

    if n < N:
        y = r[k, 0]
        x = r[k, 1]
        val = psi[bb, k, my, mx].conjugate() * z[k, my, mx]
        cuda.atomic.add(out.real, (bb, y + my, x + mx), val.real)
        cuda.atomic.add(out.imag, (bb, y + my, x + mx), val.imag)

# Cell
