"""Replace one managed result directory after a complete, verified run."""
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from .io import ROOT,load,dump


def output_path(path):
    path=Path(path).absolute()
    base=ROOT/'outputs'
    parent=base.resolve()
    if base.is_symlink() or base.is_junction() or parent.parent!=ROOT.resolve():
        raise ValueError('Project outputs/ must not be a linked directory')
    if path.is_symlink() or path.is_junction() or path.resolve().parent!=parent:
        raise ValueError('Output must be a direct, non-linked child of this project outputs/ directory')
    return path.resolve()


def check_managed(path):
    path=output_path(path)
    if not path.exists(): return path
    if not path.is_dir(): raise ValueError(f'Not a result directory: {path}')
    if any(path.iterdir()):
        marker=path/'.managed.json'
        if marker.exists():
            if load(marker).get('owner')!='fiberdino-triangle': raise ValueError(f'Unrecognized output marker: {path}')
        elif not (path/'run.json').exists() or load(path/'run.json').get('config',{}).get('representation')!='triangle_vertices_v2':
            raise ValueError(f'Refusing to replace an unrecognized directory: {path}')
    # Reject junctions/symlinks throughout before recursive removal or moving.
    for base,dirs,files in os.walk(path,followlinks=False):
        for name in dirs+files:
            entry=Path(base)/name
            if entry.is_symlink() or entry.is_junction() or not entry.resolve().is_relative_to(path):
                raise ValueError(f'Linked path in generated output: {entry}')
    return path


def remove_managed(path):
    path=check_managed(path)
    if path.exists(): shutil.rmtree(path)


@contextmanager
def output_session(path):
    """OS lock is released even after process death; stale pending data is managed."""
    out=output_path(path); out.parent.mkdir(parents=True,exist_ok=True)
    lock_path=out.parent/('.'+out.name+'.lock')
    if lock_path.is_symlink() or lock_path.is_junction(): raise ValueError('Linked output lock')
    with lock_path.open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another run is using {out}') from error
        try:
            backup=out.with_name('.'+out.name+'.previous')
            if backup.exists() and not out.exists(): check_managed(backup).rename(out)
            elif backup.exists(): remove_managed(backup)
            check_managed(out)
            yield out
        finally:
            lock.seek(0)
            if os.name=='nt': msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def prepare_output(out,predict_only,names):
    pending=out.with_name('.'+out.name+'.pending')
    remove_managed(pending)
    pending.mkdir()
    dump(pending/'.managed.json',dict(owner='fiberdino-triangle',status='building'))
    if predict_only:
        for name in names:
            for filename in ['best.pt','last.pt','history.json','training_summary.json']:
                source=out/name/filename
                if not source.is_file(): raise FileNotFoundError(f'Missing trained artifact: {source}')
                target=pending/name/filename; target.parent.mkdir(exist_ok=True)
                shutil.copy2(source,target)
        if 'full' in names and (out/'topology/fitted.pt').is_file():
            (pending/'topology').mkdir()
            shutil.copy2(out/'topology/fitted.pt',pending/'topology/fitted.pt')
        if (out/'selection_protocol.json').is_file():
            shutil.copy2(out/'selection_protocol.json',pending/'selection_protocol.json')
    return pending


def commit_output(out,pending):
    out=check_managed(out); pending=check_managed(pending)
    if pending!=out.with_name('.'+out.name+'.pending'):
        raise ValueError('Unexpected pending output path')
    verification=load(pending/'verification.json')
    if verification.get('artifact_integrity')!='PASS': raise ValueError('Output verification did not pass')
    dump(pending/'.managed.json',dict(owner='fiberdino-triangle',status='complete'))
    backup=out.with_name('.'+out.name+'.previous')
    remove_managed(backup)
    if out.exists(): out.rename(backup)
    try:
        pending.rename(out)
    except OSError:
        if backup.exists(): backup.rename(out)
        raise
    remove_managed(backup)
