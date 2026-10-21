"""Verify the offline release without changing system packages."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--system', action='store_true')
args = parser.parse_args()
manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    path = (ROOT/relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise SystemExit('Missing or unsafe package file: '+relative)
    if path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise SystemExit('Package integrity failed: '+relative)
if args.system:
    if platform.machine() not in ('aarch64', 'arm64') or sys.version_info[:2] != (3, 9):
        raise SystemExit('This offline runtime requires ARM64 Linux and Python 3.9')
    sys.path.insert(0, str(ROOT/'runtime/vendor'))
    import numpy
    import cv2
    import rknnlite
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstVideo
    Gst.init(None)
    for element in ('v4l2src', 'appsink'):
        if Gst.ElementFactory.find(element) is None:
            raise SystemExit('Missing GStreamer element: '+element)
print(json.dumps({'ok': True, 'verified_files': len(manifest['files']), 'system_checked': args.system}))
