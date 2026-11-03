import argparse, concurrent.futures, hashlib, html, re, time, urllib.request
from pathlib import Path

def download(url, dest, workers=32):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    parts=dest.parent/(dest.name+'.parts'); parts.mkdir(exist_ok=True)
    req=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(req,timeout=60) as r:
        total=int(r.headers['Content-Range'].split('/')[-1])
    size=8*1024*1024
    def fetch(i):
        start=i*size; end=min(total,start+size)-1; target=parts/f'{i:05d}'
        if target.exists() and target.stat().st_size==end-start+1: return
        for attempt in range(6):
            try:
                request=urllib.request.Request(url+('&' if '?' in url else '?')+f'part={i}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'): raise RuntimeError('Server ignored range')
                    with target.open('wb') as out:
                        while chunk:=r.read(1024*1024): out.write(chunk)
                if target.stat().st_size != end-start+1: raise RuntimeError('Short download')
                return
            except Exception as exc:
                if attempt==5: raise
                time.sleep(2+attempt)
    n=(total+size-1)//size; begin=time.time()
    print(f'Downloading {dest.name}: {total/1e6:.1f} MB in {n} parts',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for done,_ in enumerate(concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(n)]),1):
            if done%8==0 or done==n: print(f'{dest.name}: {done}/{n} parts, {time.time()-begin:.0f}s',flush=True)
    temp=dest.with_suffix(dest.suffix+'.joining')
    with temp.open('wb') as out:
        for i in range(n):
            with (parts/f'{i:05d}').open('rb') as src:
                while chunk:=src.read(8*1024*1024): out.write(chunk)
    temp.replace(dest)
    for part in parts.iterdir(): part.unlink()
    parts.rmdir()
    print('Downloaded:',dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['cub','torch']); a=p.parse_args()
    root=Path(__file__).resolve().parent
    if a.kind=='cub': download('https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz',root/'data'/'CUB_mirror.tgz')
    else:
        index=urllib.request.urlopen('https://download.pytorch.org/whl/cu128/torch/').read().decode()
        links=[html.unescape(u) for u in re.findall(r'href="([^"]+)"',index) if '2.11.0%2Bcu128-cp314-cp314-win_amd64' in u]
        url=links[-1].split('#')[0]
        dest=root/'cache'/'wheels'/'torch-2.11.0+cu128-cp314-cp314-win_amd64.whl'
        download(url,dest)
        expected=links[-1].split('#sha256=')[1]
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==expected, 'Wheel SHA256 mismatch'
        print('Wheel SHA256 verified:',expected,flush=True)
