"""Mitochondria segmentation using MitoNet or U-Net."""
import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm


def segment_with_mitonet(image_2d: np.ndarray,
                         model_config: str = "MitoNet_v1") -> np.ndarray:
    """
    Run MitoNet segmentation on a 2D EM image.
    Returns an instance label map (int32) — each unique int is one mitochondrion.
    image_2d: float32 array normalised to [0,1], shape (H, W)
    """
    from empanada.inference.engines import MultiScaleInferenceEngine
    engine = MultiScaleInferenceEngine(model_config)
    pan_seg, _ = engine.infer(image_2d)
    return pan_seg.astype(np.int32)


def segment_with_unet(image_2d: np.ndarray, model_path: str = None) -> np.ndarray:
    """
    Run U-Net binary segmentation, then connected-component labeling
    for instance separation.
    """
    import torch
    from skimage import measure, morphology
    from src.cnn_model import build_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Prepare input
    tensor = torch.from_numpy(image_2d).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    binary = (pred > 0.5).astype(np.uint8)
    # Clean up small objects
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=100)
    # Instance labeling via connected components
    instance_map = measure.label(binary).astype(np.int32)
    return instance_map


def segment_directory(input_dir: Path, output_dir: Path,
                      method: str = "mitonet", model_path: str = None):
    """Segment all preprocessed tiles in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles = sorted(input_dir.glob("*.tif"))

    for tile_path in tqdm(tiles, desc="Segmenting"):
        img = tifffile.imread(str(tile_path)).astype(np.float32)
        img = img / (img.max() + 1e-8)

        if method == "mitonet":
            mask = segment_with_mitonet(img)
        else:
            mask = segment_with_unet(img, model_path)

        out_name = tile_path.stem + "_mask.tif"
        tifffile.imwrite(str(output_dir / out_name), mask)

    print(f"Segmented {len(tiles)} tiles → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment mitochondria")
    parser.add_argument("--input", required=True, help="Input directory of preprocessed tiles")
    parser.add_argument("--output", required=True, help="Output directory for masks")
    parser.add_argument("--method", default="mitonet", choices=["mitonet", "unet"])
    parser.add_argument("--model-path", default=None, help="Path to trained U-Net weights")
    args = parser.parse_args()

    segment_directory(Path(args.input), Path(args.output), args.method, args.model_path)
