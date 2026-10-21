"""Synthetic-only checks: never open cameras or save user photos."""
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from server import CAPTURE_HEIGHT, CAPTURE_WIDTH, CaptureStore, Live


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rk-stereo-capture-test-')
        self.addCleanup(self.temporary.cleanup)
        self.store = CaptureStore(self.temporary.name)
        self.left = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH), np.uint8)
        self.right = np.full_like(self.left, 137)

    def pair(self, age=0, skew=8_000_000):
        now = time.monotonic()-age
        return (1_000_000_000, self.left, now), (1_000_000_000+skew, self.right, now)

    def test_lossless_pair_metadata_and_duplicate_rejection(self):
        data = self.store.save(self.pair(), ('left-device', 'right-device'))
        folder = Path(self.temporary.name)/data['id']
        for name, expected in [('left.png', self.left), ('right.png', self.right)]:
            decoded = cv2.imread(str(folder/name), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.array_equal(decoded, expected))
        self.assertEqual(data['skew_ms'], 8)
        self.assertFalse(data['calibrated'])
        self.assertEqual(data['square_size_reference_mm'], 24.2)
        self.assertEqual(data['calibration_target']['inner_corners'], [9, 6])
        self.assertEqual(json.loads((folder/'meta.json').read_text())['left_device'], 'left-device')
        self.assertEqual(len(self.store.entries()), 1)
        with self.assertRaisesRegex(ValueError, '已保存'):
            self.store.save(self.pair(), ('l', 'r'))
        self.assertEqual(len(self.store.entries()), 1)

    def test_reject_stale_skew_and_wrong_resolution(self):
        for pair in [self.pair(age=2), self.pair(skew=26_000_000)]:
            with self.assertRaises(ValueError):
                self.store.save(pair, ('l', 'r'))
        a, b = self.pair()
        with self.assertRaises(ValueError):
            self.store.save(((a[0], self.left[:10], a[2]), b), ('l', 'r'))
        self.assertFalse(list(Path(self.temporary.name).iterdir()))

    def test_failed_second_image_never_publishes_half_pair(self):
        with patch('server.cv2.imwrite', side_effect=[True, False]):
            with self.assertRaises(OSError):
                self.store.save(self.pair(), ('l', 'r'))
        self.assertFalse(self.store.entries())
        self.assertFalse(list(Path(self.temporary.name).iterdir()))

    def test_selector_stale_camera_and_busy_capture(self):
        live = Live.__new__(Live)
        live.frames = [deque(), deque()]
        live.store = self.store
        live.condition = threading.Condition()
        live.devices = ('l', 'r')
        live.capture_error = None
        a, b = self.pair()
        live.frames[0].append(a)
        live.frames[1].append(b)
        self.assertIsNotNone(live.select_pair())
        live.frames[1].clear()
        live.frames[1].append((b[0], b[1], time.monotonic()-2))
        self.assertIsNone(live.select_pair())
        with self.assertRaisesRegex(ValueError, '尚无'):
            live.take_photo()
        with self.store.lock:
            with self.assertRaisesRegex(ValueError, '正在保存'):
                live.take_photo()
        self.assertFalse(self.store.entries())


if __name__ == '__main__':
    unittest.main()
