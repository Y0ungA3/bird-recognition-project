import hashlib, json, tarfile
from pathlib import Path
root = Path(__file__).resolve().parent
archive = root/'data'/'CUB_200_2011.tgz'
expected = '97eceeb196236b17998738112f37df78'
h = hashlib.md5()
with archive.open('rb') as f:
    for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
print('MD5:', h.hexdigest(), flush=True)
if h.hexdigest() != expected: raise RuntimeError('Checksum mismatch; do not extract')
with tarfile.open(archive) as tar: tar.extractall(root/'data', filter='data')
data = root/'data'/'CUB_200_2011'
paths = [line.split(maxsplit=1)[1] for line in (data/'images.txt').read_text().splitlines()]
assert len(paths)==11788 and all((data/'images'/p).is_file() for p in paths)
from PIL import Image
for i, path in enumerate(paths):
    with Image.open(data/'images'/path) as im: im.verify()
    if (i+1)%2000==0: print('Verified images:',i+1,flush=True)
(root/'data'/'verification.json').write_text(json.dumps({'md5':h.hexdigest(),'images':len(paths),'all_images_verified':True},indent=2))
print('All 11788 images verified.',flush=True)
