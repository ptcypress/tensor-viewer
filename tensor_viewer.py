# tensor_viewer.py
# pip install streamlit numpy torch plotly scikit-learn

import io
import json
from typing import Tuple, Optional

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

st.set_page_config(page_title="Tensor Viewer", layout="wide")
st.title("🧭 Tensor Viewer")

# -----------------------------
# Helpers
# -----------------------------
def to_numpy(x):
    if TORCH_OK and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    return x

def load_uploaded(file) -> Optional[np.ndarray]:
    name = file.name.lower()
    data = file.read()
    bio = io.BytesIO(data)

    # Torch formats
    if TORCH_OK and (name.endswith(".pt") or name.endswith(".pth")):
        obj = torch.load(bio, map_location="cpu")
        if isinstance(obj, dict):
            # Try to find the largest tensor-like entry
            candidates = [v for v in obj.values() if isinstance(v, (torch.Tensor, np.ndarray, list))]
            if not candidates:
                st.error("No tensor-like entries found in the loaded dict.")
                return None
            arr = to_numpy(max(candidates, key=lambda v: np.array(v).size))
            return arr
        return to_numpy(obj)

    # NumPy formats
    if name.endswith(".npy"):
        return np.load(bio, allow_pickle=True)
    if name.endswith(".npz"):
        z = np.load(bio, allow_pickle=True)
        # choose the largest array inside
        keys = list(z.keys())
        if not keys:
            st.error(".npz file is empty.")
            return None
        key = max(keys, key=lambda k: z[k].size)
        return z[key]

    # JSON fallback (e.g., raw array)
    try:
        obj = json.loads(data.decode("utf-8"))
        return np.array(obj)
    except Exception:
        st.error("Unsupported file. Use .pt/.pth/.npy/.npz or JSON array.")
        return None

def normalize(arr, clip=True):
    a = arr.astype(np.float32)
    if clip:
        # robust normalization (avoid outliers)
        lo, hi = np.percentile(a, [1, 99])
        if hi > lo:
            a = np.clip(a, lo, hi)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    return a

def pick_axes_ui(shape: Tuple[int, ...]) -> Tuple[int, int, dict]:
    dims = len(shape)
    dim_labels = [f"dim{i} ({shape[i]})" for i in range(dims)]

    cols = st.columns(3)
    with cols[0]:
        x_ax = st.selectbox("X axis", list(range(dims)), format_func=lambda i: dim_labels[i], index=0)
    with cols[1]:
        y_ax = st.selectbox("Y axis", [i for i in range(dims) if i != x_ax],
                            format_func=lambda i: dim_labels[i], index=0)
    if x_ax == y_ax:
        st.stop()

    slicers = {}
    with cols[2]:
        st.write("**Slice other dims**")
    for i, n in enumerate(shape):
        if i in (x_ax, y_ax):
            continue
        slicers[i] = st.slider(f"dim{i} index (size {n})", 0, n - 1, 0)
    return x_ax, y_ax, slicers

def get_2d_slice(arr: np.ndarray, x_ax: int, y_ax: int, slicers: dict) -> np.ndarray:
    # Move desired axes to front and slice the rest
    order = [x_ax, y_ax] + [i for i in range(arr.ndim) if i not in (x_ax, y_ax)]
    v = np.transpose(arr, order)

    # Build index for remaining axes
    idx = [slice(None), slice(None)]
    for i, ax in enumerate(order[2:], start=2):
        src_dim = ax
        idx.append(slicers[src_dim])
    v2d = v[tuple(idx)]
    return v2d

def is_rgb_like(img2d_or3d: np.ndarray) -> bool:
    # Expect H x W x 3 in [0,1] or [0,255]
    return (img2d_or3d.ndim == 3 and img2d_or3d.shape[-1] in (3, 4))

# -----------------------------
# Sidebar: Source
# -----------------------------
with st.sidebar:
    st.header("Source")
    src = st.radio("Choose data source", ["Demo", "Upload"], horizontal=True)

    if src == "Demo":
        demo = st.selectbox("Demo tensor",
                            ["2D Heatmap (100x100)",
                             "RGB Image (128x128x3)",
                             "3D Volume (64x64x64)",
                             "4D Tensor (Batch x H x W x C)"])
        if demo == "2D Heatmap (100x100)":
            arr = np.random.randn(100, 100) * 0.5 + np.linspace(-1, 1, 100)[None, :]
        elif demo == "RGB Image (128x128x3)":
            x = np.linspace(0, 1, 128)
            y = np.linspace(0, 1, 128)
            xx, yy = np.meshgrid(x, y)
            r = np.sin(10*xx)*0.5+0.5
            g = yy
            b = xx
            arr = np.stack([r, g, b], axis=-1)
        elif demo == "3D Volume (64x64x64)":
            # two gaussian blobs
            z, y, x = np.mgrid[-1:1:64j, -1:1:64j, -1:1:64j]
            g1 = np.exp(-((x+0.3)**2 + (y+0.2)**2 + (z)**2)/(2*0.15**2))
            g2 = np.exp(-((x-0.2)**2 + (y-0.3)**2 + (z+0.2)**2)/(2*0.2**2))
            arr = g1 + 0.8*g2
        else:  # 4D
            arr = np.random.rand(8, 64, 64, 3)  # e.g., batch of 8 RGB images

    else:
        up = st.file_uploader("Upload .pt/.pth/.npy/.npz (or JSON array)", type=["pt","pth","npy","npz","json"])
        arr = None
        if up:
            arr = load_uploaded(up)
        if arr is None:
            st.stop()

    st.divider()
    st.caption("Basic preprocessing")
    apply_norm = st.checkbox("Normalize (robust 1–99% clip)", value=True)

# Show basic info
arr = to_numpy(arr)
if arr is None:
    st.stop()

st.write(f"**Shape:** `{arr.shape}`  |  **Dtype:** `{arr.dtype}`  |  **NDims:** `{arr.ndim}`")

# Decide default mode
mode = st.radio("Visualization", ["Auto", "Heatmap / Image (2D slice)", "3D Volume"], horizontal=True)

# -----------------------------
# 3D Volume
# -----------------------------
if (mode == "3D Volume") or (mode == "Auto" and arr.ndim == 3 and not is_rgb_like(arr)):
    if arr.ndim != 3:
        st.info("Volume view expects a 3D tensor. Switch to 2D slice mode for higher dims.")
    else:
        vol = normalize(arr) if apply_norm else arr.astype(np.float32)
        st.subheader("3D Volume")
        # Plotly Volume
        fig = go.Figure(data=go.Volume(
            value=vol, x=None, y=None, z=None, opacity=0.15,
            surface_count=18, caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=700)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 2D Slice / Image
# -----------------------------
if (mode == "Heatmap / Image (2D slice)") or (mode == "Auto" and arr.ndim != 3):
    st.subheader("2D Slice / Image")

    if arr.ndim == 1:
        # Treat as a row vector
        img2d = arr[None, :]
        x_ax = y_ax = 0
        slicers = {}
    elif arr.ndim == 2:
        img2d = arr
        x_ax, y_ax, slicers = 0, 1, {}
    else:
        x_ax, y_ax, slicers = pick_axes_ui(arr.shape)
        img2d = get_2d_slice(arr, x_ax, y_ax, slicers)

    # If last dim looks like channels, render as image
    if is_rgb_like(img2d):
        img = img2d
        if apply_norm:
            img = normalize(img, clip=False)
            if img.shape[-1] == 4:
                # keep alpha separate
                rgb = img[..., :3]
                a = img[..., 3:]
                img = np.concatenate([rgb, a], axis=-1)
        st.image(img, caption=f"Image view (shape {img.shape})", use_container_width=True)
    else:
        hm = normalize(img2d) if apply_norm else img2d.astype(np.float32)
        fig = px.imshow(hm, origin="upper", aspect="auto", color_continuous_scale="Viridis")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=700, coloraxis_colorbar=dict(title="Value"))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tips & Notes
# -----------------------------
with st.expander("Notes & Tips"):
    st.markdown(
        """
- **Axis picking**: Select which two dimensions to plot; slice the rest via sliders.
- **Images**: Any 2D slice that ends with 3 or 4 channels renders as RGB(A).
- **Normalization** helps make contrasts visible when values are tiny or dominated by outliers.
- **Volume view** expects a 3D tensor; for 4D+ volumes, pick two axes and slice the rest in the 2D mode.
- Supported uploads: **.pt / .pth / .npy / .npz / JSON array** (largest array is auto-selected from dict/npz).
"""
    )
