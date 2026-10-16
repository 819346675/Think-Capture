"""
Pi3 Utilities: Interface for Pi3 geometry reconstruction

Provides high-level API for:
- Camera pose estimation
- 3D point map generation
- Multi-view geometry

"""

import sys
import os
from contextlib import nullcontext
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch for type hints
    class torch:
        Tensor = object

# Try to import Pi3
try:
    PI3_PATH = Path(os.environ.get("PI3_PATH", "third_party/Pi3"))
    if PI3_PATH.exists():
        sys.path.insert(0, str(PI3_PATH))

    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_images_as_tensor
    PI3_AVAILABLE = True
except ImportError:
    PI3_AVAILABLE = False
    logging.warning("Pi3 not available. Install from: https://github.com/yyfz/Pi3")


class Pi3Processor:
    """
    Wrapper for Pi3 model providing simplified interface.

    Features:
    - Camera pose estimation from image sequences
    - Local point map generation
    - 3D reconstruction
    """

    def __init__(self,
                 model_checkpoint: str = None,
                 device: str = "cuda",
                 use_mock: bool = False):
        """
        Initialize Pi3 processor

        Args:
            model_checkpoint: HuggingFace model checkpoint, local path, or None (auto-detect)
            device: Device to run model on
            use_mock: Use mock implementation for testing
        """
        self.device = device
        self.use_mock = use_mock
        self.logger = logging.getLogger(__name__)

        if not use_mock and (not TORCH_AVAILABLE or not PI3_AVAILABLE):
            raise ImportError("Install PyTorch and Pi3X; see README.md. Mock mode requires use_mock=True.")
        if use_mock:
            self.logger.warning("Using MOCK Pi3 implementation")
            self.model = None
        else:
            self.logger.info("Loading Pi3 model...")

            # Auto-detect local model path
            if model_checkpoint is None:
                local_model_path = Path(__file__).parent.parent / "models" / "pi3"
                if local_model_path.exists() and any(local_model_path.iterdir()):
                    model_checkpoint = str(local_model_path)
                    self.logger.info(f"Found local model at: {model_checkpoint}")
                else:
                    model_checkpoint = "yyfz233/Pi3X"  # Default to HuggingFace

            try:
                # Set mirror environment variable if using HuggingFace
                import os
                if not Path(model_checkpoint).exists() and 'HF_ENDPOINT' not in os.environ:
                    # Can set mirror: os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    pass

                # from_pretrained supports both local and HuggingFace paths
                self.model = Pi3X.from_pretrained(model_checkpoint).to(device).eval()
                self.logger.info("Pi3 model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Pi3X checkpoint: {model_checkpoint}") from e

    def process_images(self,
                      images: List[np.ndarray],
                      return_point_cloud: bool = False) -> Dict[str, np.ndarray]:
        """
        Process image sequence to extract geometry and poses.

        Args:
            images: List of images (H, W, 3) in RGB format
            return_point_cloud: Whether to return global point cloud

        Returns:
            results: Dictionary containing:
                - 'local_point_maps': List of local point maps (H, W, 3)
                - 'camera_poses': List of 4x4 camera pose matrices
                - 'confidences': List of confidence maps (H, W)
                - 'points': Optional global point cloud (N, 3)
        """
        if self.use_mock:
            return self._mock_process(images, return_point_cloud)
        if not images:
            raise ValueError("At least one RGB image is required")

        try:
            # Convert images to tensor
            imgs_tensor = self._images_to_tensor(images)
            imgs_tensor = imgs_tensor.to(self.device)

            # Run Pi3 inference
            with torch.no_grad():
                amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                       if str(self.device).startswith('cuda') else nullcontext())
                with amp:
                    results = self.model(imgs_tensor[None])  # Add batch dim

            # Extract results
            local_points = results['local_points'][0]  # Remove batch dim
            camera_poses = results['camera_poses'][0]
            conf = results.get('conf', None)

            # Convert to numpy
            local_point_maps = [lp.float().cpu().numpy() for lp in local_points]
            poses = [p.float().cpu().numpy() for p in camera_poses]
            confidences = [c.float().cpu().numpy() if c is not None else None
                          for c in (conf[0] if conf is not None else [None] * len(poses))]

            output = {
                'local_point_maps': local_point_maps,
                'camera_poses': poses,
                'confidences': confidences
            }

            if return_point_cloud:
                points = results.get('points', None)
                if points is not None:
                    output['points'] = points[0].float().cpu().numpy()

            return output

        except Exception as e:
            raise RuntimeError("Pi3X inference failed") from e

    def _images_to_tensor(self, images: List[np.ndarray]) -> "torch.Tensor":
        """Convert list of numpy images to tensor"""
        # Pi3 requires image dimensions to be multiples of patch size (14)
        PATCH_SIZE = 14

        imgs_normalized = []
        shapes = {tuple(img.shape[:2]) for img in images}
        if len(shapes) != 1:
            raise ValueError("Pi3X input frames must have the same resolution; resize them before calling the tool.")
        for img in images:
            H, W = img.shape[:2]

            # Resize to nearest multiple of patch size
            new_H = ((H + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            new_W = ((W + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

            if new_H != H or new_W != W:
                # Resize using PIL for better quality
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((new_W, new_H), Image.Resampling.LANCZOS)
                img = np.array(pil_img)

            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            imgs_normalized.append(img)

        # Stack and convert to tensor (N, H, W, 3) -> (N, 3, H, W)
        imgs_array = np.stack(imgs_normalized, axis=0)
        imgs_tensor = torch.from_numpy(imgs_array).permute(0, 3, 1, 2).float()

        return imgs_tensor

    def _mock_process(self,
                     images: List[np.ndarray],
                     return_point_cloud: bool = False) -> Dict[str, np.ndarray]:
        """Mock processing for testing"""
        num_images = len(images)
        H, W = images[0].shape[:2]

        # Generate mock local point maps (random depth)
        local_point_maps = []
        for _ in range(num_images):
            # Generate random depth map
            depth = np.random.rand(H, W) * 5.0 + 1.0  # 1-6 meters

            # Convert to 3D coordinates (simplified pinhole model)
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            x_3d = (x_coords - W / 2) / W * depth
            y_3d = (y_coords - H / 2) / H * depth
            z_3d = depth

            point_map = np.stack([x_3d, y_3d, z_3d], axis=-1)
            local_point_maps.append(point_map)

        # Generate mock camera poses (slight translation per frame)
        camera_poses = []
        for i in range(num_images):
            pose = np.eye(4)
            pose[0, 3] = i * 0.1  # Move 0.1m per frame in x direction
            pose[2, 3] = -2.0     # 2m away from origin
            camera_poses.append(pose)

        # Generate mock confidences
        confidences = [np.random.rand(H, W).astype(np.float32) * 0.5 + 0.5
                      for _ in range(num_images)]

        output = {
            'local_point_maps': local_point_maps,
            'camera_poses': camera_poses,
            'confidences': confidences
        }

        if return_point_cloud:
            # Generate mock global point cloud
            num_points = 1000
            points = np.random.randn(num_points, 3) * 2.0
            output['points'] = points

        return output


if __name__ == "__main__":
    # Test processor
    logging.basicConfig(level=logging.INFO)

    # Create processor in mock mode
    processor = Pi3Processor(use_mock=True)

    # Test with random images
    test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                   for _ in range(4)]

    results = processor.process_images(test_images, return_point_cloud=True)

    print(f"Pi3 Processing Results:")
    print(f"  Number of frames: {len(results['local_point_maps'])}")
    print(f"  Point map shape: {results['local_point_maps'][0].shape}")
    print(f"  Camera pose shape: {results['camera_poses'][0].shape}")
    print(f"  First camera pose:\n{results['camera_poses'][0]}")
    if 'points' in results:
        print(f"  Global point cloud shape: {results['points'].shape}")
