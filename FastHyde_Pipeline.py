import os
import argparse
import numpy as np
from tqdm import tqdm
import scipy
from pysptools.material_count import HySime

# Optional readers/writers
from scipy.io import loadmat, savemat
try:
    from spectral import open_image as envi_open
    HAVE_SPECTRAL = True
except Exception:
    HAVE_SPECTRAL = False

# Denoisers
try:
    from bm3d import bm3d, BM3DStages
    HAVE_BM3D = True
except Exception:
    HAVE_BM3D = False
from skimage.restoration import denoise_wavelet

from pysptools.material_count import HySime

# ------------------------------
# IO helpers
# ------------------------------
def read_hsi(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        cube = np.load(path)
        assert cube.ndim == 3, "Expected HxWxB array in .npy"
        return cube.astype(np.float32)
    if ext == ".mat":
        md = loadmat(path)
        arrs = [v for v in md.values() if isinstance(v, np.ndarray) and v.ndim == 3]
        if not arrs:
            raise ValueError("No 3D array found in .mat")
        return arrs[0].astype(np.float32)
    if ext == ".hdr":
        if not HAVE_SPECTRAL:
            raise ImportError("Install 'spectral' for ENVI: pip install spectral")
        img = envi_open(path)
        cube = np.array(img.load())  # rows x cols x bands
        return cube.astype(np.float32)
    raise ValueError(f"Unsupported file: {path}")

def save_hsi(out_path, cube):
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".npy":
        np.save(out_path, cube.astype(np.float32))
    elif ext == ".mat":
        savemat(out_path, {"cube": cube.astype(np.float32)})
    else:
        # Default .npy
        np.save(out_path if ext else out_path + ".npy", cube.astype(np.float32))

# ------------------------------
# Substract dark frame if possible
# ------------------------------
def subtract_master_dark(cube, dark):
    if dark.shape != cube.shape:
        raise ValueError(f"Master dark shape {dark.shape} must match cube {cube.shape}")
    out = cube - dark
    np.maximum(out, 0, out=out)
    return out

# ------------------------------
# Poisson–Gaussian VST (GAT) and inverse
# Try to make the noise more gaussian
# y ~ gain * (signal + offset) + N(0, sigma_read^2)
# ------------------------------
def gat_transform(y, gain=1.0, read_sigma=0.0, offset=0.0):
    y_corr = np.maximum(y - offset, 0.0)
    inside = gain * y_corr + (3.0/8.0) * (gain**2) + (read_sigma**2)
    z = (2.0 / gain) * np.sqrt(np.maximum(inside, 0.0))
    return z

def inv_gat(z, gain=1.0, read_sigma=0.0, offset=0.0):
    x = ((gain * z) / 2.0) ** 2 - (3.0/8.0) * (gain**2) - (read_sigma**2)
    x = np.maximum(x, 0.0) / max(gain, 1e-8) + offset
    return x

# ------------------------------
# HySime-ish subspace estimation
# Comes from pysptools implementation with little tweak
# ------------------------------

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """

    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w = np.zeros((L, N), dtype=np.float64)

        # Gram matrix
        RR = np.dot(r, r.T).astype(np.float64, copy=False)
        RRi = np.linalg.pinv(RR + small * np.eye(L, dtype=np.float64))

        for i in range(L):
            # Schur complement trick without aliasing
            # XX = RRi - (RRi[:, i] RRi[i, :]) / RRi[i, i]
            denom = RRi[i, i]
            if denom <= 0:
                denom = small
            XX = RRi - np.outer(RRi[:, i], RRi[i, :]) / denom

            RRa = RR[:, i].copy()
            RRa[i] = 0.0

            beta = XX @ RRa  # shape (L,)

            # Predict band i from others: beta^T r
            # Note: beta[i] is ~0 by construction; no need to force a zero index
            Ri_hat = beta @ r  # (N,)
            w[i, :] = r[i, :] - Ri_hat

        # Diagonal noise covariance
        Rw = np.diag(np.diag((w @ w.T) / float(N)))
        Rw = 1/2 * Rw + 1/2 * Rw.T # Uncomment if want to ensure symmetricality

        return w, Rw

    y = y.T
    L, N = y.shape
    #verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T

def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape

    x = y - n

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn+np.sum(np.diag(Rx))/L/10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry,E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn,E)))
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek

def make_whitener(Rn, eps=1e-6):
    """
    Return W and Winv such that W Rn W^T ≈ I (band-wise whitening).
    Uses eigendecomposition;

    Whitten the noise so it's more i.i.d
    """
    Rn = np.asarray(Rn, dtype=np.float64)
    # Rn = 0.5 * (Rn + Rn.T)  # Uncomment if want to symmetrize
    e, U = np.linalg.eigh(Rn)

    # regularize tiny/negative eigenvalues So can be inversed and square-rooted
    floor = max(e.max(), 0.0) * eps + 1e-12
    e_cl = np.clip(e, floor, None)
    Lambda_m12 = np.diag(1.0 / np.sqrt(e_cl))
    W    = (Lambda_m12 @ U.T).astype(np.float32)   # BxB, whitener
    Winv = np.linalg.inv(W) # BxB, inverse whitener
    return W, Winv

# ------------------------------
# Denoise eigen-images (FastHyDe style)
# ------------------------------
def denoise_Z_per_image(Z, H, W, sigma_vec=None, strength=0.25):
    """
    Denoise each eigen-image with BM3D if available, else wavelet.
    Z: (k,N)  -> returns Z_deno (k,N)

    Need to normalize first. Then, using normalized version for denoising in 2D eigen images
    """
    k, N = Z.shape
    if sigma_vec is None:
        sigma_vec = np.full(k, 0.15, dtype=float)

    Z_deno = np.empty_like(Z)
    for i in range(k):
        img = Z[i].reshape(H, W)
        p1, p99 = np.percentile(img, [1, 99])
        if p99 <= p1:
            p1, p99 = (img.min(), img.max()) if img.max() > img.min() else (0.0, 1.0)
        norm = (img - p1) / max(p99 - p1, 1e-8)

        s = float(np.clip(sigma_vec[i], 1e-6, 1.0) * strength)
        if HAVE_BM3D:
            den = bm3d(norm.astype(np.float32), s, stage_arg=BM3DStages.ALL_STAGES)
        else:
            den = denoise_wavelet(norm.astype(np.float32), sigma=s,
                                  method="BayesShrink", mode="soft", rescale_sigma=True)
        Z_deno[i] = (den * (p99 - p1) + p1).reshape(-1)
    return Z_deno

def center_cube(cube):
    H, W, B = cube.shape
    X  = cube.reshape(-1, B).astype(np.float64).T      # (B, N)
    mu = X.mean(axis=1, keepdims=True)                 # (B, 1)
    Xc = X - mu                                        # (B, N)
    return Xc, mu

def noise_fraction_sigma(l_k, n_k):
    """σ_i ∝ sqrt(noise_fraction).
    For strength vector for BM3D denoising algorithm
    """
    nf = np.clip(n_k / np.maximum(l_k, n_k + 1e-12), 1e-6, 0.99)
    return np.sqrt(nf)

def reconstruct_from_Z(Ek, Z, mu, H, W, B, Winv, whitten=True):
    Xc_hat = Ek @ Z                                  # (B,N)

    if whitten:
        Xc_hat = Winv @ Xc_hat
    Xhat = (Xc_hat + mu).T.reshape(H, W, B).astype(np.float32)
    return Xhat

# ------------------------------
# Pipeline
# ------------------------------

def compute_Z_from_hysime(cube, use_est_noise=True, whitten=True):
    """
    Returns: Ek (B,k), k, Z_raw (k,N), Xc (B,N), mu (B,1),
             l_k (k,), n_k (k,) for later denoiser scheduling.
    """

    H, W, B = cube.shape
    N = H * W
    Xc, mu = center_cube(cube)  # (B,N), (B,1)

    # HySime subspace (class API accepts (H,W,B))
    Y = cube.reshape(-1, B).astype(np.float64)  # (N, B)
    w, Rw = est_noise(Y)

    if whitten:
        C_lambda, C_lambda_inv = make_whitener(Rw)
        mu = Y.mean(axis=0, keepdims=True).astype(np.float32)  # (1,B)
        Yc = (Y - mu).astype(np.float32, copy=False)  # (N,B)
        Y = Yc @ C_lambda.T  # (N,B)
        w = w @ C_lambda.T  # (N,B)
        Rw = np.eye(B, dtype=np.float32)  # whitened noise ≈ I

    k, Ek = hysime(Y, w, Rw)                  # Ek: (B, k)

    Xc = Y.T
    mu = mu.T


    # Since whitten, then E.T @ E ≈ I
    Z_raw = Ek.T @ Xc                            # (k, N)

    # Power terms for scheduling later
    Rx = (Xc @ Xc.T) / max(N - 1, 1)             # sample covariance (B,B)

    l_k = np.diag(Ek.T @ Rx @ Ek)                # total power per kept comp
    n_k = np.diag(Ek.T @ Rw @ Ek)                # noise power per kept comp

    return Ek, k, Z_raw, Xc, mu, l_k, n_k, H, W, B, C_lambda_inv


def run_pipeline(cube, master_dark=None, use_gat=False, whitten = True, gain=1.0, read_sigma=0.0, offset=0.0):
    x = cube.astype(np.float32)

    if master_dark is not None:
        x = subtract_master_dark(x, master_dark.astype(np.float32))

    if use_gat:
        x_vst = gat_transform(x, gain=gain, read_sigma=read_sigma, offset=offset)
        work = x_vst
    else:
        work = x

    # Subspace + denoise
    print("Goes Here (1)")
    Ek, k, Z_raw, Xc, mu, l_k, n_k, H, W, B, Winv = compute_Z_from_hysime(work, use_est_noise=True)
    print("Goes Here (2)")
    sig = noise_fraction_sigma(l_k, n_k)
    print("Goes Here (3)")
    Z_deno = denoise_Z_per_image(Z_raw, H, W, sigma_vec=sig, strength=0.4)
    print("Goes Here (4)")
    denoised_cube = reconstruct_from_Z(Ek, Z_deno, mu, H, W, B, Winv, whitten=True)
    print("Goes Here (5)")

    if use_gat:
        denoised_cube = inv_gat(denoised_cube, gain = gain, read_sigma=read_sigma, offset=offset)

    return denoised_cube

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="FastHyDe-style HSI denoising with optional dark subtraction & GAT")
    ap.add_argument("--cube", required=True, help="HSI cube path (.hdr/.npy/.mat)")
    ap.add_argument("--master-dark", default=None, help="Master dark cube path (same shape as cube)")
    ap.add_argument("--use-gat", action="store_true", help="Use generalized Anscombe VST (Poisson–Gaussian)")
    ap.add_argument("--whitten", action="store_true", help="Use whitten denoiser")
    ap.add_argument("--gain", type=float, default=1.0, help="Sensor gain for GAT")
    ap.add_argument("--read-sigma", type=float, default=0.0, help="Read noise std for GAT")
    ap.add_argument("--offset", type=float, default=0.0, help="Offset for GAT (dark bias after subtraction ~0)")
    ap.add_argument("--out", required=True, help="Output path (.npy or .mat). Default .npy if unknown.")
    args = ap.parse_args()

    cube = read_hsi(args.cube)
    print(f"Loaded cube: {cube.shape} (H,W,B)")

    dark = None
    if args.master_dark:
        dark = read_hsi(args.master_dark)
        print(f"Loaded master dark: {dark.shape}")

    out = run_pipeline(
        cube,
        master_dark=dark,
        use_gat=args.use_gat,
        gain=args.gain,
        read_sigma=args.read_sigma,
        offset=args.offset,
        whitten=args.whitten
    )

    save_hsi(args.out, out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
