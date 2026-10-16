"""
SpatialMemoryManager: Core class for 3D cognitive map management

This module implements real-time fusion of SAM3 (semantic/tracking) and Pi3 (geometry/pose)
outputs to maintain an incrementally updated, camera-anchored 3D cognitive map.

"""

import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available. Point cloud processing will be limited.")


@dataclass
class ObjectMemory:
    """Memory structure for a single object"""
    instance_id: int
    label: str
    world_points: np.ndarray  # Nx3 array of points in world coordinates
    world_center: np.ndarray  # 3D center in world coordinates
    scale: np.ndarray  # AABB dimensions [w, h, d]
    confidence: float
    appearance: str = ""
    num_observations: int = 1
    last_seen_frame: int = 0
    is_visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': int(self.instance_id),
            'label': self.label,
            'world_pos': self.world_center.tolist(),
            'scale': self.scale.tolist(),
            'confidence': float(self.confidence),
            'appearance': self.appearance,
            'num_observations': int(self.num_observations),
            'last_seen_frame': int(self.last_seen_frame),
            'is_occluded': not self.is_visible
        }


class SpatialMemoryManager:
    """
    Manages 3D cognitive map by fusing SAM3 and Pi3 outputs.

    Key Features:
    - Maintains global dictionary of object instances
    - Projects 2D masks to 3D using Pi3 geometry
    - Incrementally fuses new observations
    - Generates camera-anchored cognitive maps

    Attributes:
        memory (Dict[int, ObjectMemory]): Global object memory indexed by instance_id
        current_frame (int): Current frame counter
        latency_stats (Dict): Processing time statistics
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 merge_distance_threshold: float = 0.5):
        """
        Initialize SpatialMemoryManager

        Args:
            confidence_threshold: Minimum confidence for object detection
            merge_distance_threshold: Distance threshold for merging objects (meters)
        """
        self.memory: Dict[int, ObjectMemory] = {}
        self.current_frame: int = 0
        self.confidence_threshold = confidence_threshold
        self.merge_distance_threshold = merge_distance_threshold

        # Statistics
        self.latency_stats = {
            'total_update_time': 0.0,
            'total_frames_processed': 0,
            'avg_latency_per_frame': 0.0
        }

        self.logger = logging.getLogger(__name__)
        # Do not force level, let logger inherit log config (to suppress detailed logs in training)

    def project_to_world(self,
                        mask: np.ndarray,
                        local_point_map: np.ndarray,
                        camera_pose: np.ndarray) -> np.ndarray:
        """
        Project 2D mask to 3D world coordinates using Pi3 geometry.

        Args:
            mask: Binary mask (H, W) indicating object pixels
            local_point_map: Pi3 local point map (H, W, 3) in camera coordinates
            camera_pose: 4x4 camera-to-world transformation matrix

        Returns:
            world_points: Nx3 array of 3D points in world coordinates
        """
        # Extract points where mask is True
        if mask.dtype != bool:
            mask = mask > 0

        # Resize mask if dimensions don't match (Pi3 may resize images to multiples of patch size)
        mask_H, mask_W = mask.shape
        point_H, point_W = local_point_map.shape[:2]

        if mask_H != point_H or mask_W != point_W:
            from PIL import Image
            # Resize mask to match point map dimensions
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((point_W, point_H), Image.Resampling.NEAREST)
            mask = np.array(mask_pil) > 127

        # Get local 3D points for masked pixels
        local_points = local_point_map[mask]  # (N, 3)
        local_points = local_points[np.isfinite(local_points).all(axis=1)]

        if len(local_points) == 0:
            return np.array([]).reshape(0, 3)

        # Convert to homogeneous coordinates
        ones = np.ones((local_points.shape[0], 1))
        local_points_homo = np.hstack([local_points, ones])  # (N, 4)

        # Apply camera pose transformation: P_world = Pose @ P_local
        world_points_homo = (camera_pose @ local_points_homo.T).T  # (N, 4)
        world_points = world_points_homo[:, :3]  # (N, 3)

        return world_points

    def compute_aabb(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute axis-aligned bounding box center and scale.

        Args:
            points: Nx3 array of 3D points

        Returns:
            center: 3D center point
            scale: AABB dimensions [width, height, depth]
        """
        if len(points) == 0:
            return np.zeros(3), np.zeros(3)

        if OPEN3D_AVAILABLE:
            # Use Open3D for more robust AABB computation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            aabb = pcd.get_axis_aligned_bounding_box()
            center = np.array(aabb.get_center())
            scale = np.array(aabb.get_extent())
        else:
            # Fallback to numpy
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            center = (min_coords + max_coords) / 2
            scale = max_coords - min_coords

        return center, scale

    def update(self, frames_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update spatial memory with new frame observations.

        Args:
            frames_data: List of frame data, each containing:
                - 'masks': List of object masks (H, W) from SAM3
                - 'instance_ids': List of instance IDs corresponding to masks
                - 'labels': List of semantic labels
                - 'confidences': List of confidence scores
                - 'local_point_map': Pi3 local point map (H, W, 3)
                - 'camera_pose': 4x4 camera pose matrix
                - 'frame_idx': Frame index (optional)

        Returns:
            update_stats: Dictionary containing update statistics
        """
        start_time = time.time()

        num_new_objects = 0
        num_updated_objects = 0
        num_frames = len(frames_data)

        for frame_idx, frame_data in enumerate(frames_data):
            # Extract frame data
            masks = frame_data.get('masks', [])
            instance_ids = frame_data.get('instance_ids', [])
            labels = frame_data.get('labels', [])
            confidences = frame_data.get('confidences', [])
            local_point_map = frame_data.get('local_point_map')
            camera_pose = frame_data.get('camera_pose')

            if local_point_map is None or camera_pose is None:
                self.logger.warning(f"Frame {frame_idx}: Missing point map or camera pose")
                continue

            # Mark all objects as not visible in this frame initially
            for obj_id in self.memory.keys():
                self.memory[obj_id].is_visible = False

            # Process each detected object
            for mask, obj_id, label, conf in zip(masks, instance_ids, labels, confidences):
                if conf < self.confidence_threshold:
                    continue

                # Project mask to 3D world coordinates
                world_points = self.project_to_world(mask, local_point_map, camera_pose)

                if len(world_points) == 0:
                    continue

                # Check if object exists in memory
                if obj_id in self.memory:
                    # Update existing object
                    obj_mem = self.memory[obj_id]

                    # Merge new points with existing points
                    obj_mem.world_points = np.vstack([obj_mem.world_points, world_points])

                    # Downsample if too many points (keep max 10000 points)
                    if len(obj_mem.world_points) > 10000:
                        indices = np.random.choice(len(obj_mem.world_points), 10000, replace=False)
                        obj_mem.world_points = obj_mem.world_points[indices]

                    # Recompute center and scale
                    obj_mem.world_center, obj_mem.scale = self.compute_aabb(obj_mem.world_points)

                    # Update metadata
                    obj_mem.num_observations += 1
                    obj_mem.last_seen_frame = frame_data.get('frame_idx', self.current_frame)
                    obj_mem.is_visible = True
                    obj_mem.confidence = max(obj_mem.confidence, conf)  # Keep best confidence

                    num_updated_objects += 1

                else:
                    # Create new object
                    center, scale = self.compute_aabb(world_points)

                    self.memory[obj_id] = ObjectMemory(
                        instance_id=obj_id,
                        label=label,
                        world_points=world_points,
                        world_center=center,
                        scale=scale,
                        confidence=conf,
                        num_observations=1,
                        last_seen_frame=frame_data.get('frame_idx', self.current_frame),
                        is_visible=True
                    )

                    num_new_objects += 1

            self.current_frame += 1

        # Update statistics
        elapsed_time = time.time() - start_time
        self.latency_stats['total_update_time'] += elapsed_time
        self.latency_stats['total_frames_processed'] += num_frames
        self.latency_stats['avg_latency_per_frame'] = (
            self.latency_stats['total_update_time'] /
            max(1, self.latency_stats['total_frames_processed'])
        )

        update_stats = {
            'num_frames': num_frames,
            'num_new_objects': num_new_objects,
            'num_updated_objects': num_updated_objects,
            'total_objects': len(self.memory),
            'elapsed_time_ms': elapsed_time * 1000,
            'avg_time_per_frame_ms': (elapsed_time / max(1, num_frames)) * 1000
        }

        self.logger.info(f"Updated memory: {update_stats}")

        return update_stats

    def generate_cognitive_map(self, latest_camera_pose: np.ndarray) -> Dict[str, Any]:
        """
        Generate camera-anchored cognitive map.

        Transforms all object positions to be relative to the latest camera pose,
        creating a cognitive map centered on the current viewpoint.

        Args:
            latest_camera_pose: 4x4 transformation matrix of current camera pose

        Returns:
            cognitive_map: Structured JSON object conforming to schema
        """
        start_time = time.time()

        # Compute inverse of latest camera pose to transform world -> camera
        T_world_to_cam = np.linalg.inv(latest_camera_pose)

        objects_list = []
        num_visible = 0
        num_occluded = 0

        for obj_id, obj_mem in self.memory.items():
            # Transform world center to camera-relative coordinates
            world_center_homo = np.append(obj_mem.world_center, 1.0)  # Add homogeneous coord
            rel_pos_homo = T_world_to_cam @ world_center_homo
            rel_pos = rel_pos_homo[:3]

            # Build object entry
            obj_entry = {
                'id': int(obj_mem.instance_id),
                'label': obj_mem.label,
                'rel_pos': rel_pos.tolist(),
                'world_pos': obj_mem.world_center.tolist(),
                'scale': obj_mem.scale.tolist(),
                'is_occluded': not obj_mem.is_visible,
                'confidence': float(obj_mem.confidence),
                'appearance': obj_mem.appearance,
                'num_observations': int(obj_mem.num_observations),
                'last_seen_frame': int(obj_mem.last_seen_frame)
            }

            objects_list.append(obj_entry)

            if obj_mem.is_visible:
                num_visible += 1
            else:
                num_occluded += 1

        processing_time_ms = (time.time() - start_time) * 1000

        cognitive_map = {
            'timestamp': datetime.now().isoformat(),
            'camera_pose': {
                'matrix': latest_camera_pose.tolist()
            },
            'objects': objects_list,
            'statistics': {
                'total_objects': len(self.memory),
                'visible_objects': num_visible,
                'occluded_objects': num_occluded,
                'processing_time_ms': processing_time_ms
            }
        }

        return cognitive_map

    def save_cognitive_map(self, filepath: str, latest_camera_pose: np.ndarray):
        """Save cognitive map to JSON file"""
        cognitive_map = self.generate_cognitive_map(latest_camera_pose)
        with open(filepath, 'w') as f:
            json.dump(cognitive_map, f, indent=2)
        self.logger.info(f"Saved cognitive map to {filepath}")

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        return self.latency_stats.copy()

    def reset(self):
        """Reset memory and statistics"""
        self.memory.clear()
        self.current_frame = 0
        self.latency_stats = {
            'total_update_time': 0.0,
            'total_frames_processed': 0,
            'avg_latency_per_frame': 0.0
        }
        self.logger.info("Memory reset")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize manager
    manager = SpatialMemoryManager(confidence_threshold=0.5)

    # Simulate frame data (replace with actual SAM3/Pi3 outputs)
    H, W = 480, 640
    frames_data = []

    for i in range(5):
        frame_data = {
            'masks': [np.random.rand(H, W) > 0.7],  # Random mask
            'instance_ids': [1],
            'labels': ['chair'],
            'confidences': [0.85],
            'local_point_map': np.random.randn(H, W, 3),  # Random point map
            'camera_pose': np.eye(4)  # Identity pose
        }
        frames_data.append(frame_data)

    # Update memory
    stats = manager.update(frames_data)
    print(f"Update stats: {stats}")

    # Generate cognitive map
    cognitive_map = manager.generate_cognitive_map(np.eye(4))
    print(f"\nCognitive Map:")
    print(json.dumps(cognitive_map, indent=2))

    # Print latency statistics
    latency = manager.get_latency_stats()
    print(f"\nLatency Stats:")
    print(f"  Avg time per frame: {latency['avg_latency_per_frame']*1000:.2f} ms")
