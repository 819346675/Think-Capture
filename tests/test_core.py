import ast
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import SpatialMemoryManager
from agent_tool_wrapper import AgentToolWrapper, format_tool_result_for_model
from geometry.pi3_utils import Pi3Processor
from perception.sam3_wrapper import SAM3Wrapper


class GeometryTests(unittest.TestCase):
    def test_projection(self):
        manager = SpatialMemoryManager()
        points = np.array([[[1., 2., 3.], [np.nan, 0., 0.]]])
        pose = np.eye(4)
        pose[:3, 3] = [2, 0, -1]
        result = manager.project_to_world(np.ones((1, 2), dtype=bool), points, pose)
        np.testing.assert_allclose(result, [[3, 2, 2]])

    def test_aabb(self):
        center, scale = SpatialMemoryManager().compute_aabb(np.array([[0., 0., 0.], [2., 4., 6.]]))
        np.testing.assert_allclose(center, [1, 2, 3])
        np.testing.assert_allclose(scale, [2, 4, 6])

    def test_relative_frame(self):
        manager = SpatialMemoryManager()
        pose = np.eye(4)
        pose[0, 3] = 2
        manager.update([dict(masks=[np.ones((1, 1), dtype=bool)], instance_ids=[0],
                             labels=['chair'], confidences=[.9],
                             local_point_map=np.array([[[1., 0., 3.]]]),
                             camera_pose=pose, frame_idx=7)])
        obj = manager.generate_cognitive_map(pose)['objects'][0]
        np.testing.assert_allclose(obj['rel_pos'], [1, 0, 3])
        self.assertEqual(obj['last_seen_frame'], 7)

    def test_centroid_not_box_midpoint(self):
        center, scale = SpatialMemoryManager().compute_aabb(
            np.array([[0., 0., 0.], [0., 0., 0.], [3., 3., 3.]]))
        np.testing.assert_allclose(center, [1, 1, 1])
        np.testing.assert_allclose(scale, [3, 3, 3])

    @staticmethod
    def frame(labels, positions, frame_idx):
        return dict(masks=[np.ones((1, 1), dtype=bool) for _ in labels],
                    instance_ids=list(range(len(labels))), labels=labels,
                    confidences=[.9] * len(labels),
                    local_point_map=np.array([[positions]], dtype=float),
                    camera_pose=np.eye(4), frame_idx=frame_idx)

    def test_cross_view_fusion(self):
        manager = SpatialMemoryManager()
        manager.update([self.frame(['chair'], [0, 0, 2], 0),
                        self.frame(['chair'], [.2, 0, 2], 1)])
        self.assertEqual(len(manager.memory), 1)
        self.assertEqual(manager.memory[0].num_observations, 2)
        np.testing.assert_allclose(manager.memory[0].world_center, [.1, 0, 2])

    def test_fusion_preserves_distinct_labels_and_distant_instances(self):
        manager = SpatialMemoryManager()
        manager.update([self.frame(['chair'], [0, 0, 2], 0),
                        self.frame(['table'], [.1, 0, 2], 1),
                        self.frame(['chair'], [2, 0, 2], 2)])
        self.assertEqual(len(manager.memory), 3)

    def test_one_to_one_fusion(self):
        manager = SpatialMemoryManager()
        manager.update([self.frame(['chair'], [0, 0, 2], 0),
                        self.frame(['chair', 'chair'], [.1, 0, 2], 1)])
        self.assertEqual(len(manager.memory), 2)
        self.assertEqual(sorted(o.num_observations for o in manager.memory.values()), [1, 2])

    def test_missing_dependencies_fail_closed(self):
        with patch('geometry.pi3_utils.PI3_AVAILABLE', False):
            with self.assertRaises(ImportError):
                Pi3Processor(use_mock=False)
        with patch('perception.sam3_wrapper.SAM3_AVAILABLE', False):
            with self.assertRaises(ImportError):
                SAM3Wrapper(use_mock=False)

    def test_inference_error_never_calls_mock(self):
        processor = Pi3Processor(use_mock=True)
        processor.use_mock = False
        with patch.object(processor, '_images_to_tensor', side_effect=RuntimeError('test failure')):
            with patch.object(processor, '_mock_process') as mock:
                with self.assertRaises(RuntimeError):
                    processor.process_images([np.zeros((14, 14, 3), dtype=np.uint8)])
                mock.assert_not_called()
        sam = SAM3Wrapper(use_mock=True)
        sam.use_mock = False
        with patch.object(sam, '_mock_segment') as mock:
            with self.assertRaises(RuntimeError):
                sam.segment_image(np.zeros((14, 14, 3), dtype=np.uint8))
            mock.assert_not_called()

    def test_mock_pipeline_and_frame_ids(self):
        np.random.seed(42)
        wrapper = AgentToolWrapper(use_mock=True, device='cpu')
        result = wrapper.process_images([f'mock_view_{i}.jpg' for i in range(6)], ['chair'])
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['is_mock'])
        self.assertEqual(result['statistics']['frames_processed'], 4)
        self.assertEqual(result['camera_info']['selected_frame_indices'], [0, 1, 3, 5])
        np.testing.assert_allclose(result['camera_info']['all_poses'][0], np.eye(4))


class FormattingTests(unittest.TestCase):
    def test_opencv_camera_direction(self):
        pose = np.eye(4)
        next_pose = np.eye(4)
        next_pose[:3, 3] = [0, -1, 2]
        result = dict(status='success', objects=[], camera_info=dict(
            all_poses=[pose.tolist(), next_pose.tolist()], selected_frame_indices=[0, 5]))
        text = format_tool_result_for_model(result, question='Where is the camera positioned relative to me?')
        self.assertIn('in front up', text)
        self.assertIn('Image 6 camera', text)
        self.assertIn('positive Y and negative Z', text)

    def test_object_vertical_direction(self):
        result = dict(status='success', camera_info={}, objects=[
            dict(label='lamp', world_pos=[0, -1, 2], scale=[1, 1, 1], confidence=.9),
            dict(label='table', world_pos=[0, 0, 2], scale=[1, 1, 1], confidence=.9),
        ])
        text = format_tool_result_for_model(result, question='Where is the lamp relative to the table?')
        self.assertIn('LAMP is up of TABLE', text)


class RewardTests(unittest.TestCase):
    def test_reward_weights(self):
        # Test the pure reward functions without importing the GPU trainer.
        path = Path(__file__).resolve().parents[1] / 'ours/training/grpo_stage2/train.py'
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'parse_answer', 'compute_rewards'}]
        namespace = {'re': re, 'List': list}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        rewards = namespace['compute_rewards']
        outputs = ['<think>Reason.</think><answer>A</answer>',
                   '<think>Reason.</think><answer>B</answer>', 'Answer: A',
                   '<answer>A</answer>', '<think>Unclosed<answer>A</answer>']
        self.assertEqual(rewards(outputs, 'A', 'positive'), [1.5, .5, 1., 1., 1.])
        self.assertEqual(rewards(outputs, 'A', 'negative'), [2.5, .5, 2., 2., 2.])
        with self.assertRaises(ValueError):
            rewards(outputs, 'A', 'invalid')


if __name__ == '__main__':
    unittest.main()
