"""Download official DINOv2 small/register code and weights into this project only."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent':'FiberDINO-research/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    weights = ROOT/'weights/dinov2_vits14_reg4_pretrain.pth'
    provenance = ROOT/'weights/provenance.json'
    code = ROOT/'third_party/dinov2'
    if provenance.exists() and weights.exists() and (code/'dinov2/models/vision_transformer.py').exists():
        info = json.loads(provenance.read_text(encoding='utf-8'))
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=info['weights_sha256']:
            raise RuntimeError('Cached weights hash mismatch')
        print('Official DINOv2 assets already available and hash checked.', flush=True)
        return
    revision = json.loads(fetch('https://api.github.com/repos/facebookresearch/dinov2/commits/main'))['sha']
    print('Official DINOv2 revision: '+revision, flush=True)
    archive_url = f'https://codeload.github.com/facebookresearch/dinov2/zip/{revision}'
    archive = zipfile.ZipFile(io.BytesIO(fetch(archive_url)))
    hashes = {}
    # Only the backbone implementation and its license are needed.
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts[1:]
        if not parts or member.is_dir():
            continue
        relative = PurePosixPath(*parts)
        name = relative.as_posix()
        if not (name in {'LICENSE','README.md','dinov2/__init__.py'} or
                name.startswith(('dinov2/layers/','dinov2/models/','dinov2/hub/'))):
            continue
        target = (code/Path(*parts)).resolve()
        if not target.is_relative_to(code.resolve()) or '..' in parts or ':' in name:
            raise ValueError('Unsafe archive member')
        target.parent.mkdir(parents=True, exist_ok=True)
        content = archive.read(member)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    print('Downloading official ViT-S/14 register weights (~84 MiB)...', flush=True)
    content = fetch(WEIGHTS_URL)
    if len(content)<10_000_000:
        raise ValueError('Unexpected weight file size')
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(content)
    provenance.write_text(json.dumps(dict(repository='https://github.com/facebookresearch/dinov2',
                                          revision=revision, weights_url=WEIGHTS_URL,
                                          weights_sha256=hashlib.sha256(content).hexdigest(),
                                          code_sha256=hashes, license='Apache-2.0'), indent=2), encoding='utf-8')
    print(f'Download complete: {len(content):,} bytes', flush=True)


if __name__=='__main__':
    main()
