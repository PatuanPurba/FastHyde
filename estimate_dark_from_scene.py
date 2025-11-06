# save as estimate_dark_from_scene.py
import os, numpy as np
from scipy.ndimage import sobel
from scipy.io import loadmat, savemat

def read_cube(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext == ".mat":
        md = loadmat(path)
        arrs = [v for v in md.values() if isinstance(v, np.ndarray) and v.ndim == 3]
        if not arrs: raise ValueError("No 3D array in .mat")
        return arrs[0].astype(np.float32)
    else:
        raise ValueError("Use .npy or .mat here for brevity")

def save_arr(path, arr):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy": np.save(path, arr.astype(np.float32))
    else: savemat(path if ext else path + ".mat", {"cube": arr.astype(np.float32)})

def estimate_dark_from_scene(cube, mode="per_column", pct=2.0):
    H, W, B = cube.shape
    # Low-gradient mask (avoid edges/texture)
    gx = sobel(cube.mean(axis=2), axis=1)
    gy = sobel(cube.mean(axis=2), axis=0)
    grad = np.hypot(gx, gy)
    gthr = np.percentile(grad, 30)  # keep the flattest ~30%
    flat = grad <= gthr

    # Also avoid brightest pixels
    bright = cube.mean(axis=2) > np.percentile(cube.mean(axis=2), 70)
    mask = flat & (~bright)

    est = np.zeros_like(cube, dtype=np.float32)
    for b in range(B):
        sb = cube[..., b]
        mb = mask
        if mode == "global":
            off = np.percentile(sb[mb], pct)
            est[..., b] = off
        elif mode == "per_column":
            # robust per-column median from masked pixels
            col_off = np.zeros(W, np.float32)
            for j in range(W):
                vals = sb[mb[:, j], j]
                col_off[j] = np.percentile(vals, pct) if vals.size else 0.0
            est[..., b] = col_off[None, :]
        else:
            raise ValueError("mode must be 'global' or 'per_column'")
    # Don’t add negative bias
    est = np.clip(est, 0, None)
    return est

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--out-dark", default="estimated_dark.npy")
    ap.add_argument("--mode", choices=["global","per_column"], default="per_column")
    ap.add_argument("--pct", type=float, default=2.0)
    args = ap.parse_args()
    X = read_cube(args.cube)
    Dhat = estimate_dark_from_scene(X, mode=args.mode, pct=args.pct)
    save_arr(args.out_dark, Dhat)
    print("Saved:", args.out_dark, "shape", Dhat.shape)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--out-dark", default="estimated_dark.npy")
    ap.add_argument("--mode", choices=["global","per_column"], default="per_column")
    ap.add_argument("--pct", type=float, default=2.0)
    args = ap.parse_args()
    X = read_cube(args.cube)
    Dhat = estimate_dark_from_scene(X, mode=args.mode, pct=args.pct)
    save_arr(args.out_dark, Dhat)
    print("Saved:", args.out_dark, "shape", Dhat.shape)