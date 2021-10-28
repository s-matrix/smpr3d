from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt, numpy as np
from PIL import Image
import tqdm
import matplotlib as mpl, sys
import mpl_toolkits.mplot3d.axes3d as mpl3d
from scipy.ndimage.filters import gaussian_filter as blur


def plot_3d_wavefunction(
    ax,
    intensity,
    phase,
    cmap=mpl.cm.gist_rainbow,
    azdeg=315,
    altdeg=45,
    satmin=0.25,
    satmax=0.75,
    alpha=0.9,
    flatenning=5,
):
    """On axis ax, make a 3d colorplot of a wave function with intensity and
    phase in numpy arrays.
    Parameters
    ax - axis on which to plot the image
    intensity - Intensity of the complex wave function
    phase - Phase of complex wave function
    cmap - color map for plotting phase
    azdeg - azimuthal angle of light source
    altdeg - angle of light source from z axis
    alpha - transperancy of plot

    """

    from matplotlib.colors import LightSource

    def renormalize(array, newmin=0, newmax=1):
        oldmin, oldmax = [np.amin(array), np.amax(array)]
        return (array - oldmin) * (newmax - newmin) / (oldmax - oldmin)

    # Get grids of x and y
    X, Y = np.meshgrid(*[np.arange(intensity.shape[i]) for i in [0, 1]])

    # Height is intensity
    Z = renormalize(intensity)

    # Setup light source
    ls = LightSource(azdeg, altdeg)

    # rgb =
    # C = rgb
    C = ls.blend_hsv(
        cmap(phase % (2 * np.pi) / (2 * np.pi)),
        renormalize(intensity).reshape([*inten.shape, 1]),
        hsv_min_sat=1.0,
        hsv_max_sat=1.0,
        hsv_min_val=satmin,
        hsv_max_val=satmax,
    )

    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=C,
        rcount=X.shape[1],
        ccount=X.shape[0],
        alpha=alpha,
        shade=True,
    )
    ax.set_axis_off()
    ax.set_zlim(0, flatenning)


def top_hat_filter(img, radius):
    """Applies a top hat filter out to a certain radius (in fractions of the
    grid) in Fourier space"""
    fftimg = np.fft.fft2(np.asarray(img, dtype=np.complex))
    kx, ky = np.meshgrid(np.fft.fftfreq(img.shape[0]), np.fft.fftfreq(img.shape[1]))
    fftimg[kx ** 2 + ky ** 2 > radius] = 0
    return np.fft.ifft2(fftimg)


def fourier_interpolate_2d(ain, npiy_, npix_, norm=True):
    """ain - input numpy array, npiy_ number of y pixels in
    interpolated image,  npix_ number of x pixels in interpolated
    image. Perfoms a fourier interpolation on array ain. Not yet
    set up to also perform downsampling"""
    from numpy.fft import fft2, fftshift, ifft2

    # Make input complex, fourier transform it and calculate
    # dimensions
    ain_ = fftshift(fft2(np.asarray(ain, dtype=np.complex)))
    npiy, npix = ain_.shape
    # Make output array of appropriate shape
    aout = np.zeros((npiy_, npix_), dtype=np.complex)

    def make_indices(pix, pix_):
        def indic(pixl, pixl_):
            even = pixl % 2 == 0
            even_ = pixl_ % 2 == 0
            if (not even) and even_:
                up_, low_ = (pixl_ // 2 + pixl // 2 + 1, pixl_ // 2 - pixl // 2)
            elif (not even) == (not even_):
                up_, low_ = (pixl_ // 2 + pixl // 2, pixl_ // 2 - pixl // 2)
            else:
                up_, low_ = (pixl_ // 2 + pixl // 2, pixl_ // 2 - pixl // 2)
            up, low = (pixl, 0)
            return up, low, up_, low_

        if pix <= pix_:
            return indic(pix, pix_)
        else:
            up, low, up_, low_ = indic(pix_, pix)
            return up_, low_, up, low

    # Calculate appropriate array indices
    upperx, lowerx, upperx_, lowerx_ = make_indices(npix, npix_)
    uppery, lowery, uppery_, lowery_ = make_indices(npiy, npiy_)

    # place input fft on new grid, padded with zeros
    aout[lowery_:uppery_, lowerx_:upperx_] = ain_[lowery:uppery, lowerx:upperx]

    # Fourier transform result with appropriate normalization
    aout = ifft2(fftshift(aout))
    if norm:
        aout *= 1.0 * npiy_ * npix_ / npix / npiy
    # if(norm): aout*=np.sqrt(1.0*npiy_*npix_/npix/npiy)

    # Return correct array type
    if (
        str(ain.dtype) in ["float64", "float32", "float", "f"]
        or str(ain.dtype)[1] == "f"
    ):
        return np.real(aout)
    else:
        return aout


def crop(array, newsize):
    y, x = array.shape
    y_, x_ = newsize
    return array[(y - y_) // 2 : (y + y_) // 2, (x - x_) // 2 : (x + x_) // 2]


def renormalize(array, newmin=0, newmax=1):
    oldmin, oldmax = [np.amin(array), np.amax(array)]
    return (array - oldmin) * (newmax - newmin) / (oldmax - oldmin)


def submatsum(data, n, m):
    # return a matrix of shape (n,m)
    bs = data.shape[0] // n, data.shape[1] // m  # blocksize averaged over
    return np.reshape(
        np.array(
            [
                np.sum(
                    data[k1 * bs[0] : (k1 + 1) * bs[0], k2 * bs[1] : (k2 + 1) * bs[1]]
                )
                for k1 in range(n)
                for k2 in range(m)
            ]
        ),
        (n, m),
    )


def bin_array(a, bins):
    return np.sum(
        a.reshape(bins[0], a.shape[0] // bins[0], bins[1], a.shape[1] // bins[1]),
        axis=(0, 2),
    )


indx = [3, 1, 0, 2, 6]
for i in tqdm.tqdm(range(5)):
    for j in range(2):
        # ax = fig.add_subplot(111+i*10,projection = '3d')
        inten = np.asarray(Image.open("S{1}_{0}_inten.tiff".format(indx[i] + 1, j + 1)))
        phase = np.asarray(Image.open("S{1}_{0}_phase.tiff".format(indx[i] + 1, j + 1)))

        c = np.sqrt(inten) * np.exp(1j * phase)
        c = top_hat_filter(c, 0.1)
        
        c = crop(fourier_interpolate_2d(c, 256, 256), [256 * 3 // 4, 256 * 3 // 4])
        inten = np.square(np.abs(c))
        phase = np.angle(c)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d", azim=-10 + i * 5)
        plot_3d_wavefunction(
            ax,
            inten,
            phase,
            cmap=plt.get_cmap("gist_rainbow"),
            azdeg=270,
            altdeg=10,
            satmin=0.8,
            satmax=1.0,
            alpha=1.0,
            flatenning=15,
        )
        plt.tight_layout()
        fig.savefig("Smatrix_{0}_{1}.png".format(i, j), dpi=500, transparent=True)
        plt.show(block=True)
        plt.clf()
