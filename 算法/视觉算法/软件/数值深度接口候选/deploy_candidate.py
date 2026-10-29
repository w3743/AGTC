"""Board-only, two-phase install/restore for the explicitly listed app files."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

APP = Path('/opt/rk-lightstereo/app')
BACKUPS = APP.parent / 'backups'
FILES = ('server.py', 'depth_wire.py', 'depth_adapter.py', 'hardware_video.py',
         'native_roi.py', 'roi_depth.py', 'index.html', 'far.html')
BASELINE = {
    'server.py': '554c30e9c179989074246335d9e347aba6062caf95f0204267b8021c87c0bf6e',
    'fast_depth.py': 'df48bae5e63a5882d51cfbbf418b754189cf81acbe648cc575d6950c66d109e4',
    'calibration.npz': 'fc0ee7b44fe56c397d0abab6cf94140d85c5017793c13656cbeb09a2ed9c2a1e',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restart():
    subprocess.run(['systemctl', 'restart', 'rk-lightstereo.service'], check=True, timeout=40)


def install_atomic(source, target, mode):
    temporary = target.with_name('.' + target.name + '.optimization-tmp')
    with temporary.open('wb') as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(mode)
    temporary.replace(target)


def restore(backup, manifest):
    for name in FILES:
        state = manifest['files'][name]
        target = APP / name
        if state['existed']:
            saved = backup / 'app' / name
            if sha(saved) != state['before_sha256']:
                raise RuntimeError('Backup hash mismatch: ' + name)
            install_atomic(saved, target, state['mode'])
        elif target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError('Unexpected restore target: ' + str(target))
            target.unlink()
    restart()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'apply', 'rollback'))
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--backup', type=Path, required=True)
    args = p.parse_args()
    backup = args.backup.resolve()
    if APP.resolve() != APP or backup.parent != BACKUPS.resolve():
        p.error('Expected the real app directory and one direct child of its backups directory')
    if args.mode == 'prepare':
        for name, expected in BASELINE.items():
            if sha(APP / name) != expected:
                raise RuntimeError('Active baseline differs; inspect before install: ' + name)
        source = (args.source / 'app').resolve()
        manifest = {'scope': 'listed app files only; no calibration/model/runtime/service configuration changes',
                    'source': str(source), 'files': {}}
        for name in FILES:
            new = source / name
            if not new.is_file() or new.is_symlink():
                raise RuntimeError('Missing regular source file: ' + name)
            if name.endswith('.py'):
                ast.parse(new.read_text(encoding='utf-8'), filename=name, feature_version=(3, 9))
            old = APP / name
            if old.is_symlink() or (old.exists() and not old.is_file()):
                raise RuntimeError('Unexpected active file type: ' + name)
            manifest['files'][name] = dict(existed=old.exists(),
                before_sha256=sha(old) if old.exists() else None,
                after_sha256=sha(new), mode=(old.stat().st_mode & 0o777) if old.exists() else 0o644)
        backup.mkdir(parents=True, exist_ok=False)
        (backup / 'app').mkdir()
        for name, state in manifest['files'].items():
            if state['existed']:
                shutil.copy2(APP / name, backup / 'app' / name)
                if sha(backup / 'app' / name) != state['before_sha256']:
                    raise RuntimeError('Backup verification failed: ' + name)
        (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        archive = backup.with_suffix('.tar.gz')
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(backup, arcname=backup.name)
        print(json.dumps({'prepared': str(backup), 'archive': str(archive),
                          'archive_sha256': sha(archive), 'files': manifest['files']}))
        return
    manifest = json.loads((backup / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['files']) != set(FILES):
        raise RuntimeError('Backup manifest targets do not match this installer')
    if args.mode == 'rollback':
        restore(backup, manifest)
        print(json.dumps({'rolled_back': str(backup)}))
        return
    source = Path(manifest['source'])
    for name in FILES:
        state = manifest['files'][name]
        if sha(source / name) != state['after_sha256']:
            raise RuntimeError('Candidate changed after preparation: ' + name)
        target = APP / name
        if (target.exists() != state['existed']
                or (state['existed'] and sha(target) != state['before_sha256'])):
            raise RuntimeError('Active file changed after backup: ' + name)
    try:
        for name, state in manifest['files'].items():
            install_atomic(source / name, APP / name, state['mode'])
            if sha(APP / name) != state['after_sha256']:
                raise RuntimeError('Installed hash mismatch: ' + name)
        restart()
    except Exception:
        restore(backup, manifest)
        raise
    print(json.dumps({'installed': list(FILES), 'rollback_backup': str(backup)}))


if __name__ == '__main__':
    main()
