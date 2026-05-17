"""Image preprocessing for EM mitochondria images."""
import argparse
import numpy as np
import tifffile
from skimage import exposure
from pathlib import Path
from typing import Tuple


def load_tif_stack(path: str | Path) -> np.ndarray:
    """Load a single .tif or multi-page .tif stack. Returns (Z, H, W) or (H, W)."""
    return tifffile.imread(str(path))


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] float32."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def apply_clahe(img: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a 2D grayscale image [0,1]."""
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)


def tile_image(img: np.ndarray, tile_size: int = 512,
               overlap: int = 64) -> Tuple[list, list]:
    """
    Tile a 2D image into overlapping patches.
    Returns list of (patch, (row_start, col_start)) tuples.
    """
    H, W = img.shape[:2]
    stride = tile_size - overlap
    patches, coords = [], []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            patch = img[r:r_end, c:c_end]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                pad_h = tile_size - patch.shape[0]
                pad_w = tile_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="reflect")
            patches.append(patch)
            coords.append((r, c))
    return patches, coords


def preprocess_slice(img_2d: np.ndarray, tile_size: int = 512) -> np.ndarray:
    """Full preprocessing pipeline for a single 2D EM slice."""
    img = normalize_image(img_2d)
    img = apply_clahe(img)
    return img


def preprocess_stack(input_path: Path, output_dir: Path,
                     tile_size: int = 512, overlap: int = 64):
    """Preprocess an entire .tif stack: normalize, CLAHE, tile, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = load_tif_stack(input_path)

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]  # treat single image as 1-slice stack

    for z_idx in range(stack.shape[0]):
        img = preprocess_slice(stack[z_idx], tile_size)
        patches, coords = tile_image(img, tile_size, overlap)
        for p_idx, (patch, (r, c)) in enumerate(zip(patches, coords)):
            fname = f"slice_{z_idx:04d}_tile_{p_idx:04d}_r{r}_c{c}.tif"
            tifffile.imwrite(str(output_dir / fname), (patch * 65535).astype(np.uint16))

    print(f"Preprocessed {stack.shape[0]} slices → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EM images")
    parser.add_argument("--input", required=True, help="Path to input .tif file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    preprocess_stack(Path(args.input), Path(args.output), args.tile_size, args.overlap)
