import numpy as np
from skimage.metrics import structural_similarity as ssim

def _to_float(x):
    x = np.asarray(x)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32)
    return x

def psnr_per_band(clean, test, data_range="band"):
    H, W, B = clean.shape
    psnr = np.zeros(B, dtype=np.float64)
    for b in range(B):
        c = clean[..., b]; t = test[..., b]
        mse = np.mean((c - t)**2)
        if mse == 0:
            psnr[b] = np.inf
            continue
        if data_range == "band":
            dr = float(c.max() - c.min())
        elif data_range == "global":
            dr = float(clean.max() - clean.min())
        else:   # assume reflectance [0,1]
            dr = 1.0
        dr = max(dr, 1e-8)
        psnr[b] = 20*np.log10(dr) - 10*np.log10(mse)
    return psnr

def ssim_per_band(clean, test, data_range="band"):
    H, W, B = clean.shape
    out = np.zeros(B, dtype=np.float64)
    if data_range == "band":
        ranges = [float(clean[...,b].max() - clean[...,b].min()) for b in range(B)]
    elif data_range == "global":
        gr = float(clean.max() - clean.min()); ranges = [gr]*B
    else:
        ranges = [1.0]*B
    for b in range(B):
        c = clean[..., b]; t = test[..., b]
        dr = max(ranges[b], 1e-8)
        out[b] = ssim(c, t, data_range=dr, gaussian_weights=True, use_sample_covariance=False)
    return out

def rmse_per_band(clean, test):
    err = clean - test
    H, W, B = err.shape
    rmse = np.sqrt(np.mean(err**2, axis=(0,1)))
    return rmse, float(np.sqrt(np.mean(err**2)))

def sam_map(clean, test, eps=1e-12):
    """Return SAM per pixel in degrees."""
    C = clean.reshape(-1, clean.shape[2]).astype(np.float64)
    T = test.reshape(-1, test.shape[2]).astype(np.float64)
    dot = np.sum(C*T, axis=1)
    nc  = np.linalg.norm(C, axis=1) + eps
    nt  = np.linalg.norm(T, axis=1) + eps
    cosang = np.clip(dot / (nc*nt), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang.reshape(clean.shape[0], clean.shape[1])

def ergas(clean, test, scale=1.0):
    """ERGAS with scale=1 for denoising (no resolution change)."""
    H, W, B = clean.shape
    rmse_b = np.sqrt(np.mean((clean - test)**2, axis=(0,1)))
    mean_b = np.clip(np.mean(clean, axis=(0,1)), 1e-8, None)
    val = 100.0/scale * np.sqrt(np.mean((rmse_b / mean_b)**2))
    return float(val)

def evaluate_hsi(clean, denoised, mask=None, data_range="band"):
    C = _to_float(clean); D = _to_float(denoised)
    assert C.shape == D.shape and C.ndim == 3, "Shapes must match (H,W,B)."
    if mask is not None:
        m = mask.astype(bool)
        C = C[m].reshape(-1, C.shape[2]).reshape(m.sum(), 1, C.shape[2])
        D = D[m].reshape(-1, D.shape[2]).reshape(m.sum(), 1, D.shape[2])
        # reshape back to pseudo image with 1 column to reuse functions
        H = m.sum(); W = 1
        C = C.reshape(H, W, -1); D = D.reshape(H, W, -1)

    ps = psnr_per_band(C, D, data_range=data_range)
    ss = ssim_per_band(C, D, data_range=data_range)
    rmse_b, rmse_all = rmse_per_band(C, D)
    sam_deg = sam_map(C, D)
    out = {
        "mPSNR_dB": float(np.mean(ps[np.isfinite(ps)])),
        "mSSIM": float(np.mean(ss)),
        "mean_SAM_deg": float(np.mean(sam_deg)),
        "median_SAM_deg": float(np.median(sam_deg)),
        "ERGAS": ergas(C, D, scale=1.0),
        "RMSE_overall": rmse_all,
        "PSNR_per_band": ps,
        "SSIM_per_band": ss,
        "RMSE_per_band": rmse_b,
        "SAM_map": sam_deg,  # 2D map for visualization if mask is None
    }
    return out

denoised = np.load("denoised_mat_cuprite.npy")
denoised = np.transpose(denoised, (1, 2, 0))
clean = np.load("cuprite512.npy")
den = denoised

H,W,B = clean.shape
Xb = den.reshape(-1, B)
Yb = clean.reshape(-1, B)

# Per-band slopes/offsets via closed-form LS
mX = Xb.mean(axis=0)
mY = Yb.mean(axis=0)
varX = Xb.var(axis=0) + 1e-12
covXY = ((Xb - mX).T @ (Yb - mY)) / (Xb.shape[0]-1)
a_b = covXY.diagonal() / varX               # slopes per band (length B)
b_b = mY - a_b * mX                         # offsets per band

print("Slope per-band:  mean=%.4g  std=%.4g  min=%.4g  max=%.4g" %
      (a_b.mean(), a_b.std(), a_b.min(), a_b.max()))
print("Offset per-band: mean=%.4g  std=%.4g  min=%.4g  max=%.4g" %
      (b_b.mean(), b_b.std(), b_b.min(), b_b.max()))


out = evaluate_hsi(clean, denoised)

print(out)

