"""
SAM3 Wrapper: Interface for Segment Anything Model 3

Provides high-level API for:
- Object instance segmentation
- Instance tracking across frames
- Semantic labeling

"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import SAM3
try:
    # Add SAM3 codebase to path if available
    SAM3_PATH = Path(os.environ.get("SAM3_PATH", "third_party/sam3"))
    if SAM3_PATH.exists():
        sys.path.insert(0, str(SAM3_PATH))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except Exception as e:
    SAM3_AVAILABLE = False
    logging.warning(f"SAM3 not available due to import error: {e}. "
                    "Please ensure sam3 and its dependencies are installed from: https://github.com/facebookresearch/sam3")


class SAM3Wrapper:
    """
    Wrapper for SAM3 model providing simplified interface.

    Features:
    - Automatic instance segmentation
    - Text-prompted segmentation
    - Instance ID tracking
    """

    def __init__(self,
                 model_checkpoint: str = None,
                 device: str = "cuda",
                 use_mock: bool = False,
                 confidence_threshold: float = 0.5):
        """
        Initialize SAM3 wrapper

        Args:
            model_checkpoint: HuggingFace model checkpoint, local path, or None (auto-detect)
            device: Device to run model on ('cuda' or 'cpu')
            use_mock: Use mock implementation for testing
        """
        self.device = device
        self.use_mock = use_mock
        self.logger = logging.getLogger(__name__)

        if not use_mock and (not TORCH_AVAILABLE or not SAM3_AVAILABLE):
            raise ImportError("Install SAM3 and its dependencies; see README.md. Mock mode requires use_mock=True.")

        if self.use_mock:
            self.logger.warning("Using MOCK SAM3 implementation (use_mock=True)")
            self.model = None
            self.processor = None
        else:
            self.logger.info("Loading SAM3 model...")

            # Auto-detect local model path
            if model_checkpoint is None:
                local_model_path = Path(__file__).parent.parent / "models" / "sam3"
                if local_model_path.exists() and any(local_model_path.iterdir()):
                    model_checkpoint = str(local_model_path)
                    self.logger.info(f"Found local model at: {model_checkpoint}")
                else:
                    model_checkpoint = "facebook/sam3"  # Default HuggingFace

            try:
                # Local check
                checkpoint_path = None
                if Path(model_checkpoint).exists():
                    if Path(model_checkpoint).is_dir():
                        for ckpt_file in ['sam3.pt']:
                            ckpt_path = Path(model_checkpoint) / ckpt_file
                            if ckpt_path.exists():
                                checkpoint_path = str(ckpt_path)
                                self.logger.info(f"Found checkpoint: {checkpoint_path}")
                                break
                        if checkpoint_path is None:
                            raise ValueError(f"No checkpoint file found in {model_checkpoint}")
                    else:
                        checkpoint_path = model_checkpoint
                    self.logger.info(f"Loading from local checkpoint: {checkpoint_path}")
                    self.model = build_sam3_image_model(
                        checkpoint_path=checkpoint_path,
                        load_from_HF=False, device=device
                    )
                else:
                    if model_checkpoint != "facebook/sam3":
                        raise FileNotFoundError(f"SAM3 checkpoint not found: {model_checkpoint}")
                    # HuggingFace load
                    self.logger.info(f"Loading from HuggingFace: {model_checkpoint}")
                    import os
                    if 'HF_ENDPOINT' not in os.environ:
                        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
                    self.model = build_sam3_image_model(
                        checkpoint_path=None,
                        load_from_HF=True, device=device
                    )

                self.model.to(device)
                self.model.eval()
                self.processor = Sam3Processor(
                    self.model, device=device,
                    confidence_threshold=confidence_threshold
                )
                self.confidence_threshold = confidence_threshold
                self.logger.info(f"SAM3 model loaded successfully (confidence_threshold={confidence_threshold})")
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM3 checkpoint: {model_checkpoint}") from e

    def segment_image(self,
                     image: np.ndarray,
                     text_prompt: Optional[str] = None,
                     return_masks: bool = True) -> Dict[str, Any]:
        """
        Segment objects in an image.

        Args:
            image: Input image (H, W, 3) in RGB format
            text_prompt: Optional text prompt for targeted segmentation
            return_masks: Whether to return full masks

        Returns:
            results: Dictionary containing:
                - 'masks': List of binary masks (H, W)
                - 'instance_ids': List of instance IDs
                - 'labels': List of semantic labels
                - 'confidences': List of confidence scores
                - 'boxes': List of bounding boxes [x1, y1, x2, y2]
        """
        if self.use_mock:
            return self._mock_segment(image, text_prompt)

        try:
            # Convert numpy to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Set image in processor
            inference_state = self.processor.set_image(pil_image)

            # Run segmentation
            if text_prompt:
                # Use text prompt for targeted segmentation
                output = self.processor.set_text_prompt(
                    prompt=text_prompt,
                    state=inference_state
                )
            else:
                # Automatic segmentation: use "visual" as text prompt to detect all objects
                output = self.processor.set_text_prompt(
                    prompt="visual",
                    state=inference_state
                )

            # Extract results
            masks = output.get("masks", None)
            boxes = output.get("boxes", None)
            scores = output.get("scores", None)

            if masks is None or len(masks) == 0:
                return {
                    'masks': [],
                    'instance_ids': [],
                    'labels': [],
                    'confidences': [],
                    'boxes': []
                }

            # Convert masks to list of 2D binary masks
            mask_list = []
            for mask in masks:
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                # Ensure 2D mask
                if mask.ndim > 2:
                    mask = mask.squeeze()
                if mask.ndim == 2:
                    mask_list.append(mask.astype(bool))
                else:
                    # If still not 2D, take first channel
                    mask_list.append(mask[0].astype(bool) if mask.ndim > 2 else mask.astype(bool))

            num_objects = len(mask_list)
            instance_ids = list(range(num_objects))
            labels = [f"object_{i}" for i in instance_ids]

            # Convert scores
            if scores is not None:
                if torch.is_tensor(scores):
                    confidences = scores.cpu().numpy().tolist()
                else:
                    confidences = [float(s) for s in scores]
            else:
                confidences = [0.9] * num_objects

            # Convert boxes
            if boxes is not None:
                if torch.is_tensor(boxes):
                    boxes_list = boxes.cpu().numpy().tolist()
                else:
                    boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
            else:
                boxes_list = []

            results = {
                'masks': mask_list,
                'instance_ids': instance_ids,
                'labels': labels,
                'confidences': confidences,
                'boxes': boxes_list
            }

            return results

        except Exception as e:
            raise RuntimeError("SAM3 segmentation failed") from e

    def segment_batch(self,
                     images: List[np.ndarray],
                     text_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Segment multiple images in batch.

        Args:
            images: List of input images
            text_prompt: Optional text prompt

        Returns:
            results_list: List of segmentation results
        """
        results_list = []
        for img in images:
            results = self.segment_image(img, text_prompt)
            results_list.append(results)
        return results_list

    def _mock_segment(self, image: np.ndarray, text_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Mock segmentation for testing"""
        H, W = image.shape[:2]

        # Generate 2-3 random masks
        num_objects = np.random.randint(2, 4)
        masks = []
        instance_ids = []
        labels = []
        confidences = []
        boxes = []

        for i in range(num_objects):
            # Random rectangular mask
            x1 = np.random.randint(0, W // 2)
            y1 = np.random.randint(0, H // 2)
            x2 = np.random.randint(W // 2, W)
            y2 = np.random.randint(H // 2, H)

            mask = np.zeros((H, W), dtype=bool)
            mask[y1:y2, x1:x2] = True

            masks.append(mask)
            instance_ids.append(i)
            labels.append(['chair', 'table', 'sofa'][i % 3])
            confidences.append(0.7 + np.random.rand() * 0.3)
            boxes.append([x1, y1, x2, y2])

        return {
            'masks': masks,
            'instance_ids': instance_ids,
            'labels': labels,
            'confidences': confidences,
            'boxes': boxes
        }


if __name__ == "__main__":
    # Test wrapper
    logging.basicConfig(level=logging.INFO)

    # Create wrapper in mock mode
    wrapper = SAM3Wrapper(use_mock=True)

    # Test with random image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    results = wrapper.segment_image(test_image)

    print(f"Segmentation Results:")
    print(f"  Number of objects: {len(results['instance_ids'])}")
    print(f"  Instance IDs: {results['instance_ids']}")
    print(f"  Labels: {results['labels']}")
    print(f"  Confidences: {results['confidences']}")
