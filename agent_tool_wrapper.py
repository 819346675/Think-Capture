"""
Agent Tool Wrapper: Orchestrates Pi3+SAM3 pipeline for on-demand 3D detail extraction

This module provides a high-level interface for the agent-based tool calling pipeline.
Instead of full scene reconstruction, it only processes semantic targets specified by the model.

"""

import time
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import sys

# Ensure core.py can be imported from anywhere
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from core import SpatialMemoryManager
from perception.sam3_wrapper import SAM3Wrapper
from geometry.pi3_utils import Pi3Processor


class AgentToolWrapper:
    """
    High-level wrapper for agent-based tool calling.

    Workflow:
    1. Receive tool_request with semantic_targets
    2. Run Pi3 for 3D reconstruction
    3. Run SAM3 for targeted segmentation (only specified objects)
    4. Extract 3D details for matched objects
    5. Return tool_result
    """

    def __init__(self,
                 sam3_checkpoint: str = None,
                 pi3_checkpoint: str = None,
                 use_mock: bool = False,
                 device: str = "cuda"):
        """
        Initialize agent tool wrapper.

        Args:
            sam3_checkpoint: Path to SAM3 model checkpoint
            pi3_checkpoint: Path to Pi3 model checkpoint
            use_mock: If True, use mock models for testing
            device: Device for model inference
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.use_mock = use_mock

        # Initialize models
        self.logger.info("Initializing SAM3...")
        self.sam3 = SAM3Wrapper(
            model_checkpoint=sam3_checkpoint or os.environ.get("SAM3_CHECKPOINT"),
            confidence_threshold=0.3,  # Default, can be overridden
            device=device,
            use_mock=use_mock
        )

        self.logger.info("Initializing Pi3...")
        self.pi3 = Pi3Processor(
            model_checkpoint=pi3_checkpoint or os.environ.get("PI3_CHECKPOINT"),
            device=device,
            use_mock=use_mock
        )

        # Memory manager for object tracking
        self.memory_manager = SpatialMemoryManager(
            confidence_threshold=0.3,
            merge_distance_threshold=0.5
        )

        self.logger.info("Agent tool wrapper initialized successfully")
        # Pi3 VRAM limit: max frames (covers most 2-4 image scenarios, avoids 57GB+ OOM)
        self.max_pi3_frames = 4

    def process_images(self,
                      image_paths: List[str],
                      semantic_targets: List[str],
                      confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Process images with targeted semantic segmentation.

        Args:
            image_paths: List of image file paths
            semantic_targets: List of semantic labels to detect (e.g., ["chair", "table"])
            confidence_threshold: Minimum confidence for detection

        Returns:
            tool_result: Dictionary containing detected 3D objects
        """
        start_time = time.time()

        try:
            # Reset memory for new session
            self.memory_manager.reset()

            # Update SAM3 confidence threshold
            self.sam3.confidence_threshold = confidence_threshold
            self.memory_manager.confidence_threshold = confidence_threshold
            if self.sam3.processor is not None:
                self.sam3.processor.confidence_threshold = confidence_threshold

            # Step 1: Load all images
            images = []
            for img_path in image_paths:
                if self.use_mock and not Path(img_path).exists():
                    # Create dummy image in mock mode
                    image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                else:
                    image = Image.open(img_path).convert("RGB")
                    image_np = np.array(image)
                images.append(image_np)

            self.logger.info(f"Loaded {len(images)} images")

            # Limit frames sent to Pi3 to avoid OOM; temporally uniformly sample at most max_pi3_frames
            indices = list(range(len(images)))
            if len(images) > self.max_pi3_frames:
                indices = np.linspace(0, len(images) - 1, self.max_pi3_frames, dtype=int)
                images = [images[i] for i in indices]
                self.logger.info(f"Downsampled to {len(images)} frames for Pi3 (max_pi3_frames={self.max_pi3_frames})")

            # Step 2: Pi3 - Process frames for 3D geometry and camera poses
            self.logger.info("Running Pi3 for 3D reconstruction...")
            pi3_results = self.pi3.process_images(images, return_point_cloud=False)
            local_point_maps = pi3_results['local_point_maps']
            camera_poses = pi3_results['camera_poses']
            # Anchor all geometry to the first selected camera (OpenCV axes).
            world_to_first = np.linalg.inv(camera_poses[0])
            camera_poses = [world_to_first @ pose for pose in camera_poses]

            self.logger.info(f"  Pi3 generated {len(local_point_maps)} point maps")

            # Step 3: Process each frame with SAM3
            frames_data = []
            instance_id_counter = 0

            for frame_idx, (image_np, local_point_map, camera_pose) in enumerate(
                zip(images, local_point_maps, camera_poses)
            ):
                self.logger.info(f"Processing frame {frame_idx+1}/{len(images)} with SAM3...")

                # SAM3 - Targeted segmentation for each semantic target
                all_masks = []
                all_instance_ids = []
                all_labels = []
                all_confidences = []

                for target_label in semantic_targets:
                    # Segment this specific semantic target
                    seg_result = self.sam3.segment_image(
                        image_np,
                        text_prompt=target_label
                    )

                    masks = seg_result['masks']
                    confidences = seg_result['confidences']

                    # Add each detected instance
                    for mask, conf in zip(masks, confidences):
                        all_masks.append(mask)
                        all_instance_ids.append(instance_id_counter)
                        all_labels.append(target_label)
                        all_confidences.append(conf)
                        instance_id_counter += 1

                self.logger.info(f"  Frame {frame_idx+1}: Detected {len(all_masks)} objects "
                               f"(targets: {semantic_targets})")

                # Prepare frame data for memory update
                frame_data = {
                    'masks': all_masks,
                    'instance_ids': all_instance_ids,
                    'labels': all_labels,
                    'confidences': all_confidences,
                    'local_point_map': local_point_map,
                    'camera_pose': camera_pose,
                    'frame_idx': int(indices[frame_idx])
                }
                frames_data.append(frame_data)

            # Step 3: Update memory with all frames
            self.logger.info("Updating spatial memory...")
            update_stats = self.memory_manager.update(frames_data)

            # Step 4: Generate cognitive map (using the first frame as reference, because questions are often contextually based on image 1)
            first_pose = frames_data[0]['camera_pose']
            cognitive_map = self.memory_manager.generate_cognitive_map(first_pose)

            # Save all camera poses for subsequent analysis
            all_camera_poses = [fd['camera_pose'].tolist() for fd in frames_data]

            # Step 5: Format tool_result
            tool_result = {
                'status': 'success',
                'is_mock': self.use_mock,
                'objects': cognitive_map['objects'],
                'camera_info': {
                    'reference_frame': 1,  # use the first frame as reference
                    'total_frames': len(image_paths),
                    'selected_frame_indices': [int(i) for i in indices],
                    'all_poses': all_camera_poses
                },
                'statistics': {
                    'total_objects_detected': len(cognitive_map['objects']),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'frames_processed': len(images),
                    'semantic_targets_requested': semantic_targets,
                    'update_stats': update_stats
                }
            }

            self.logger.info(f"Tool execution completed: {len(cognitive_map['objects'])} objects detected")

            # VRAM cleanup: release Pi3/SAM3 intermediate results
            del local_point_maps, camera_poses, frames_data, images
            import gc
            gc.collect()

            return tool_result

        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'objects': [],
                'error_message': str(e),
                'statistics': {
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'frames_processed': 0
                }
            }

    def execute_tool_request(self,
                            image_paths: List[str],
                            tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool request from model.

        Args:
            image_paths: List of image file paths
            tool_request: Tool request dictionary conforming to schema

        Returns:
            tool_result: Tool execution result
        """
        # Validate tool_request
        if tool_request.get('pipeline') != 'pi3+sam3':
            return {
                'status': 'error',
                'objects': [],
                'error_message': f"Unsupported pipeline: {tool_request.get('pipeline')}"
            }

        semantic_targets = tool_request.get('semantic_targets', [])
        if not semantic_targets:
            return {
                'status': 'error',
                'objects': [],
                'error_message': "No semantic_targets specified"
            }

        confidence_threshold = tool_request.get('confidence_threshold', 0.3)

        # Execute
        return self.process_images(
            image_paths=image_paths,
            semantic_targets=semantic_targets,
            confidence_threshold=confidence_threshold
        )


def detect_question_type(question: str) -> str:
    """
    Infer question type from the question text, without relying on gold category.
    Priority: cam-cam > object-object > camera-view.
    Returns: "cam-cam" | "object-object" | "camera-view"
    """
    import re
    q_lower = question.lower()
    # 1) cam-cam
    cam_cam_patterns = [
        "camera positioned relative to me", "camera.*relative to me",
        "where is the camera.*when taking the second", "when taking the second photo.*camera",
        "second photo.*where.*camera", "assuming i am taking the first photo",
        "rotate.*around the (x|y|z) axis", "move in the (positive|negative) (x|y|z)",
        "rotate (by|a) (positive|negative) angle",
    ]
    for p in cam_cam_patterns:
        if re.search(p, q_lower):
            return "cam-cam"
    # 2) object-object (exclude "relative to you/camera")
    is_person_or_camera = (
        "relative to you" in q_lower or "relative to me" in q_lower
        or "relative to him" in q_lower or "in relation to you" in q_lower
        or "from the perspective of" in q_lower
        or "when you took" in q_lower or "where was .* in relation to you" in q_lower
    )
    if not is_person_or_camera:
        object_object_patterns = [
            "in comparison to", "with respect to", "positioned relative",
            "location of .* in comparison", "in which direction is .* from ",
            "direction of .* from ", "which direction is .* from ",
            "relative to the ", " is to the (north|south|east|west)",
        ]
        for p in object_object_patterns:
            if re.search(p, q_lower) or p in q_lower:
                return "object-object"
    return "camera-view"


def format_tool_result_for_model(tool_result: Dict[str, Any], max_per_category: int = 2,
                                  num_frames: int = None, question: str = None, category: str = None) -> str:
    """
    Select 3D template by question type: object-object / camera-view / cam-cam.
    Routing: if category is provided, prioritize it; otherwise use detect_question_type(question).
    Multistep reasoning, motion perception, and geometric shape use camera-view or default.
    """
    import numpy as np
    from collections import defaultdict

    if tool_result['status'] != 'success':
        return f"Tool execution failed: {tool_result.get('error_message', 'Unknown error')}"

    objects = tool_result.get('objects', [])
    camera_info = tool_result.get('camera_info', {})
    all_camera_poses = camera_info.get('all_poses', [])
    frame_ids = camera_info.get('selected_frame_indices', list(range(len(all_camera_poses))))

    # Prioritize template by category
    q_type = None
    if category:
        if "object-object" in category:
            q_type = "object-object"
        elif "cam-cam" in category:
            q_type = "cam-cam"
        elif "cam-object" in category or "cam-scene" in category:
            q_type = "camera-view"
        elif "scene-scene" in category:
            # Empirical logic: scene-scene works better with camera-view, unify to camera-view
            q_type = "camera-view"
        elif "object-scene" in category:
            # A/B suggestion: default object-scene to camera-view
            q_type = "camera-view"
    if q_type is None:
        q_type = detect_question_type(question) if question else "camera-view"

    if q_type == "cam-cam":
        if not all_camera_poses:
            return "No camera pose data available."
    else:
        if not objects:
            return "No relevant objects detected in the images."

    # ---- Helper: Camera facing description (for cam-cam enhancement) ----
    def describe_facing_direction(pose):
        """Convert camera facing direction to human-readable description (world coordinates)."""
        arr = np.array(pose)
        # OpenCV: +X right, +Y down, +Z forward
        forward = arr[:3, 2]
        forward_norm = forward / (np.linalg.norm(forward) + 1e-8)

        # Project to horizontal plane (x-z) and vertical direction
        horiz = np.array([forward_norm[0], 0, forward_norm[2]])
        horiz_len = np.linalg.norm(horiz)
        pitch_deg = np.degrees(np.arcsin(np.clip(-forward_norm[1], -1, 1)))

        dirs = []
        if horiz_len > 0.1:
            horiz = horiz / horiz_len
            # Use atan2 for horizontal yaw (relative to +Z axis as 0 degrees)
            angle = np.degrees(np.arctan2(horiz[0], horiz[2]))  # 0°=+Z, 90°=+X
            # Convert to eight directions
            angle = angle % 360
            compass = [
                (337.5, 360, "+Z direction"), (0, 22.5, "+Z direction"),
                (22.5, 67.5, "+X/+Z direction"), (67.5, 112.5, "+X direction"),
                (112.5, 157.5, "+X/-Z direction"), (157.5, 202.5, "-Z direction"),
                (202.5, 247.5, "-X/-Z direction"), (247.5, 292.5, "-X direction"),
                (292.5, 337.5, "-X/+Z direction"),
            ]
            for lo, hi, label in compass:
                if lo <= angle < hi:
                    dirs.append(label)
                    break

        if pitch_deg > 15:
            dirs.append("tilted upward")
        elif pitch_deg < -15:
            dirs.append("tilted downward")

        return ", ".join(dirs) if dirs else "unresolved"

    def compute_yaw_change(pose_from, pose_to):
        """Calculate horizontal yaw angle and direction from pose_from to pose_to."""
        arr_from, arr_to = np.array(pose_from), np.array(pose_to)
        fwd_from = arr_from[:3, 2]
        fwd_to = arr_to[:3, 2]
        # Project to horizontal plane
        fwd_from_h = np.array([fwd_from[0], 0, fwd_from[2]])
        fwd_to_h = np.array([fwd_to[0], 0, fwd_to[2]])
        len_from = np.linalg.norm(fwd_from_h)
        len_to = np.linalg.norm(fwd_to_h)
        if len_from < 1e-6 or len_to < 1e-6:
            return 0, "no significant horizontal rotation"
        fwd_from_h /= len_from
        fwd_to_h /= len_to
        cos_angle = np.clip(np.dot(fwd_from_h, fwd_to_h), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        # Cross product for left/right (y component > 0 means ccw = left turn)
        cross_y = fwd_from_h[0] * fwd_to_h[2] - fwd_from_h[2] * fwd_to_h[0]
        if angle_deg < 5:
            return angle_deg, "nearly same direction"
        direction = "turned right" if cross_y < 0 else "turned left"
        return angle_deg, direction

    objects_by_label = defaultdict(list)
    for obj in objects:
        objects_by_label[obj['label']].append(obj)
    filtered_objects = []
    for label, objs in objects_by_label.items():
        sorted_objs = sorted(objs, key=lambda x: x['confidence'], reverse=True)
        filtered_objects.extend(sorted_objs[:max_per_category])

    def get_camera_position(pose):
        arr = np.array(pose)
        return arr[:3, 3]
    def get_camera_relative_to_camera(pose_i, pose_j):
        pi, pj = np.array(pose_i), np.array(pose_j)
        vec = pj[:3, 3] - pi[:3, 3]
        dist = float(np.linalg.norm(vec))
        if dist < 1e-4:
            return "at same position", dist
        cam_right, cam_forward = pi[:3, 0], pi[:3, 2]
        cam_up = -pi[:3, 1]
        right_proj = np.dot(vec, cam_right)
        forward_proj = np.dot(vec, cam_forward)
        up_proj = np.dot(vec, cam_up)
        dirs = []
        if forward_proj > 0.15: dirs.append("in front")
        elif forward_proj < -0.15: dirs.append("behind")
        if right_proj > 0.15: dirs.append("right")
        elif right_proj < -0.15: dirs.append("left")
        if up_proj > 0.15: dirs.append("up")
        elif up_proj < -0.15: dirs.append("down")
        return " ".join(dirs) if dirs else "nearby", dist

    # Helper: calculate relative direction of object (from camera view)
    def get_direction_from_camera(obj_pos, cam_pose):
        pose_arr = np.array(cam_pose)
        cam_pos = pose_arr[:3, 3]
        cam_right = pose_arr[:3, 0]
        cam_forward = pose_arr[:3, 2]
        obj_vec = np.array(obj_pos) - cam_pos

        right_proj = np.dot(obj_vec, cam_right)
        forward_proj = np.dot(obj_vec, cam_forward)

        dirs = []
        if forward_proj > 0.2:
            dirs.append("in front")
        elif forward_proj < -0.2:
            dirs.append("behind")
        if right_proj > 0.2:
            dirs.append("right")
        elif right_proj < -0.2:
            dirs.append("left")

        return " ".join(dirs) if dirs else "nearby"

    # Helper: calculate relative direction of object A to B (from camera view, for ViewSpatial)
    def get_relative_direction(pos_a, pos_b):
        """Calculate direction of A relative to B, using smaller thresholds for precision"""
        diff = np.array(pos_a) - np.array(pos_b)
        dirs = []

        # Use smaller thresholds (0.05m=5cm), more sensitive to small objects
        # Y-axis (up/down) uses tightest threshold, e.g. "object on table" is common
        threshold_xy = 0.05  # horizontal threshold
        threshold_y = 0.03   # vertical threshold
        threshold_z = 0.1    # depth threshold (relatively loose)

        # X-axis: pos=right, neg=left (from camera)
        if diff[0] > threshold_xy:
            dirs.append("right")
        elif diff[0] < -threshold_xy:
            dirs.append("left")

        # OpenCV Y-axis: positive down, negative up
        if diff[1] > threshold_y:
            dirs.append("down")
        elif diff[1] < -threshold_y:
            dirs.append("up")

        # Z-axis: pos=front/far, neg=back/near
        if diff[2] > threshold_z:
            dirs.append("front")
        elif diff[2] < -threshold_z:
            dirs.append("back")

        if not dirs:
            return "at same position as"
        return "-".join(dirs) + " of"

    lines = []

    if q_type == "cam-cam":
        lines.append("Camera positions:")
        poses = [np.array(p) for p in all_camera_poses]
        for i, pose in enumerate(poses):
            pos = get_camera_position(pose)
            lines.append(f"  - Image {frame_ids[i]+1} camera: world position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
        lines.append("\n(Positions above use OpenCV axes: +X right, +Y down, +Z forward. Option-axis displacements below use +X right, +Y up, -Z forward.)")

        # Enhancement: camera facing direction
        lines.append("\nCamera facing directions:")
        for i, pose in enumerate(poses):
            facing = describe_facing_direction(pose)
            lines.append(f"  - Image {frame_ids[i]+1} camera is facing: {facing}")

        if len(poses) >= 2:
            p0 = poses[0]
            cam_right = p0[:3, 0]
            cam_up = -p0[:3, 1]
            cam_forward = p0[:3, 2]  # OpenCV forward; convert signs only when naming option axes.

            lines.append("\nRelative camera positions (from your view = Image 1):")
            for j in range(1, len(poses)):
                rel_dir, dist = get_camera_relative_to_camera(poses[0], poses[j])
                lines.append(f"  - From your (Image 1) view, Image {frame_ids[j]+1} camera is {dist:.2f}m {rel_dir} of you.")

            # Align with options: use "positive/negative X/Y/Z" (MMSI: +Y up, -Z forward, +X right)
            lines.append("\nDisplacement in camera coordinates (Image 1 frame: +X right, +Y up, -Z forward):")
            for j in range(1, len(poses)):
                vec = poses[j][:3, 3] - poses[0][:3, 3]
                dx = float(np.dot(vec, cam_right))
                dy = float(np.dot(vec, cam_up))
                # Along viewing direction: cam_forward points to -Z, dot>0 means towards -Z
                dz_along_view = float(np.dot(vec, cam_forward))
                parts = []
                if abs(dx) > 0.05: parts.append("positive X" if dx > 0 else "negative X")
                if abs(dy) > 0.05: parts.append("positive Y" if dy > 0 else "negative Y")
                if abs(dz_along_view) > 0.05: parts.append("negative Z" if dz_along_view > 0 else "positive Z")  # forward = -Z
                if not parts:
                    parts.append("negligible")
                lines.append(f"  - Image 1 → Image {frame_ids[j]+1}: movement in " + " and ".join(parts) + ".")

            # Rotation: clarify axis and pos/neg
            lines.append("\nCamera rotation between views:")
            for j in range(1, len(poses)):
                yaw_deg, yaw_desc = compute_yaw_change(poses[0], poses[j])
                axis_note = " (yaw = rotation around Y axis; turning right = negative Y rotation, turning left = positive Y rotation)"
                lines.append(f"  - From Image 1 to Image {frame_ids[j]+1}: {yaw_desc} (~{yaw_deg:.0f}°){axis_note}")

        # Enhancement: if objects present, add object direction per camera view
        if filtered_objects:
            lines.append("\nKey objects from each camera view:")
            for obj in filtered_objects[:4]:  # Max 4 key objects
                label = obj['label'].upper()
                world_pos = obj.get('world_pos', obj.get('rel_pos', [0,0,0]))
                for i, pose in enumerate(all_camera_poses):
                    direction = get_direction_from_camera(world_pos, pose)
                    cam_pos = get_camera_position(np.array(pose))
                    dist_to_cam = float(np.linalg.norm(np.array(world_pos) - cam_pos))
                    lines.append(f"  - {label} from Image {frame_ids[i]+1}: {direction}, {dist_to_cam:.1f}m away")

        return "\n".join(lines)

    if q_type == "object-object":
        lines.append("3D Object Positions:")

        # Output coordinates for each object
        for obj in filtered_objects:
            label = obj['label'].upper()
            world_pos = obj.get('world_pos', obj.get('rel_pos', [0,0,0]))
            size = max(obj['scale'])
            lines.append(f"• {label}: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}), size ~{size:.1f}m")

        # Calculate and output relative object positions
        if len(filtered_objects) >= 2:
            lines.append("\nSpatial Relationships:")
            for i, obj_a in enumerate(filtered_objects):
                for obj_b in filtered_objects[i+1:]:
                    pos_a = obj_a.get('world_pos', obj_a.get('rel_pos', [0,0,0]))
                    pos_b = obj_b.get('world_pos', obj_b.get('rel_pos', [0,0,0]))
                    rel_dir = get_relative_direction(pos_a, pos_b)
                    lines.append(f"• {obj_a['label'].upper()} is {rel_dir} {obj_b['label'].upper()}")

        # Enhancement: object direction and distance from camera view
        if all_camera_poses:
            lines.append("\nFrom your camera perspective:")
            for obj in filtered_objects:
                label = obj['label'].upper()
                world_pos = obj.get('world_pos', obj.get('rel_pos', [0,0,0]))
                per_frame_dirs = []
                for i, pose in enumerate(all_camera_poses):
                    direction = get_direction_from_camera(world_pos, pose)
                    cam_pos = get_camera_position(np.array(pose))
                    dist_to_cam = float(np.linalg.norm(np.array(world_pos) - cam_pos))
                    per_frame_dirs.append(f"Image {frame_ids[i]+1}: {direction}, {dist_to_cam:.1f}m away")
                lines.append(f"  {label} → " + " | ".join(per_frame_dirs))

    else:
        # camera-view format: valid for MMSI. Distinguish scene/area to avoid affecting cam-object/cam-scene
        lines.append("3D Analysis:")
        is_scene_relation = category and ("scene-scene" in category or "object-scene" in category)
        if is_scene_relation:
            lines.append("(Labels ending with ' AREA' denote scene/region; others are objects.)")
        for obj in filtered_objects:
            label = (_display_label(obj.get('label', '')) if is_scene_relation else obj.get('label', '').upper()) or obj['label'].upper()
            world_pos = obj.get('world_pos', obj.get('rel_pos', [0,0,0]))
            scale = obj.get('scale', [0,0,0])
            size = max(scale) if scale else 0
            last_seen = obj.get('last_seen_frame', 0)
            lines.append(f"\n{label} (size ~{size:.1f}m, seen in image {last_seen+1}):")
            if all_camera_poses:
                for i, pose in enumerate(all_camera_poses):
                    direction = get_direction_from_camera(world_pos, pose)
                    lines.append(f"  - From image {frame_ids[i]+1} view: {direction}")
    return "\n".join(lines)


# Scene/area targets: add " AREA" to distinguish from objects
_SCENE_LABELS = frozenset({
    "bathroom", "kitchen", "stairwell", "corridor", "dining room", "bedroom", "living room", "office",
    "billiards area", "parking lot", "fireplace area", "sofa lounge area", "well-lit area", "dining area",
    "room with piano", "access area", "doorway", "display area", "storage area", "passage", "hallway",
})


def _display_label(label: str) -> str:
    """Keep objects caps; add ' AREA' for scenes/regions."""
    if not label:
        return ""
    raw = label.strip().lower()
    if raw in _SCENE_LABELS or raw.endswith(" area") or " area" in raw:
        base = raw.replace(" area", "").strip()
        return (base if base else raw).upper() + " AREA"
    return label.upper()
