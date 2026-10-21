"""Shared runtime paths; each demo keeps only its most recent result."""
import fcntl
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('ROBOTICS_HOME', str(Path.home()/'robotics-lab')))
RUNTIME.mkdir(parents=True, exist_ok=True)

def lock_run(name):
    handle = (RUNTIME/(name+'.lock')).open('w')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('同类实验已在运行，请先在原窗口按 Ctrl+C。')
    return handle

def output_dir(name, custom=None):
    path = Path(custom) if custom else ROOT/'results'/name
    if path.is_symlink():
        raise RuntimeError('结果目录不能是符号链接。')
    path.mkdir(parents=True, exist_ok=True)
    if custom and any(path.iterdir()):
        raise RuntimeError('--output 必须使用新的空目录，以免混合不同实验。')
    # Remove only this application's known generated files, never arbitrary data.
    names = {'acceptance.json','rgb.png','depth.png','map.db','map.db-wal',
             'map.db-shm','map.db-journal','uwb_positions.csv','visual_positions.csv',
             'gazebo.log','bridge.log','camera_tf.log','visual_odometry.log',
             'rtabmap.log','map_viewer.log','px4.log'}
    if not custom:
        for item in path.iterdir():
            if item.name in names and item.is_file():
                item.unlink()
    return path
