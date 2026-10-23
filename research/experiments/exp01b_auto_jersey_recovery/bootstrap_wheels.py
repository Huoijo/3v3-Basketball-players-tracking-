"""Download compatible PyPI wheels in parallel, verify their published SHA256.

Transport helper for a slow connection; installation is separate and offline.
Uses the existing Python 3.12 Torch/NumPy environment without replacing it.
"""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.specifiers import SpecifierSet

OUT=Path('/tmp/trackid3x3-exp01b-wheels');OUT.mkdir(exist_ok=True)
PACKAGES={'easyocr':'1.7.2','pandas':'2.3.3','scipy':'1.17.0','scikit-image':'0.25.2','python-bidi':None,'Shapely':None,'pyclipper':None,'ninja':None,'imageio':None,'tifffile':None,'lazy-loader':None,'python-dateutil':None,'pytz':None,'tzdata':None,'six':None}
TAGS={t:i for i,t in enumerate(sys_tags())}


def fetch_package(item):
    package,version=item
    suffix=f'/{version}' if version else ''
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}{suffix}/json',timeout=60) as r:meta=json.load(r)
    options=[]
    for f in meta['urls']:
        if f['packagetype']!='bdist_wheel':continue
        if f.get('requires_python') and not SpecifierSet(f['requires_python']).contains('.'.join(map(str,sys.version_info[:3]))):continue
        *_,tags=parse_wheel_filename(f['filename'])
        ranks=[TAGS[t] for t in tags if t in TAGS]
        if ranks:options.append((min(ranks),f['filename'],f))
    assert options,package
    f=min(options)[2]
    dest=OUT/f['filename'];size=f['size'];chunk=2**21
    def piece(i):
        a=i*chunk;b=min(size,(i+1)*chunk)-1
        p=OUT/(f['filename']+f'.part{i}')
        if p.exists() and p.stat().st_size==b-a+1:return p
        for attempt in range(3):
            try:
                existing=p.stat().st_size if p.exists() else 0
                start=a+existing
                req=urllib.request.Request(f['url'],headers={'Range':f'bytes={start}-{b}'})
                with urllib.request.urlopen(req,timeout=120) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{b}/{size}'
                    with p.open('ab') as w:
                        while block:=r.read(2**18):w.write(block)
                assert p.stat().st_size==b-a+1
                return p
            except Exception:
                if attempt==2:raise
    if not dest.exists():
        with ThreadPoolExecutor(4) as pool:parts=list(pool.map(piece,range((size+chunk-1)//chunk)))
        with dest.open('wb') as w:
            for p in parts:w.write(p.read_bytes())
        for p in parts:p.unlink()
    with dest.open('rb') as r:actual=hashlib.file_digest(r,'sha256').hexdigest()
    assert actual==f['digests']['sha256'],package
    print(package,meta['info']['version'],'VERIFIED',size,flush=True)
    return dict(package=package,version=meta['info']['version'],filename=f['filename'],sha256=actual)


if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool:
        results=[f.result() for f in as_completed([pool.submit(fetch_package,p) for p in PACKAGES.items()])]
    (Path(__file__).resolve().parent/'wheel_sources.json').write_text(json.dumps(results,indent=2)+'\n')
