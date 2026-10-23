"""Fetch byte-identical EasyOCR weights with range requests; verify official MD5.

Mirror transport only. These are the original model files, not new models.
Official checksum source: JaidedAI/EasyOCR v1.7.2 easyocr/config.py.
"""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT=Path(__file__).resolve().parent
MODELS=[('craft_mlt_25k','7fcca5cbab94e60f54b1c9c930f162cc24bf43b6','2f8227d2def4037cdb3b34389dcf9ec1'),('english_g2','17cdb173ef73b32eb6e9d1270f33b30154b03908','5864788e1821be9e454ec108d61b887d')]


def main():
    records=[]
    for name,revision,md5 in MODELS:
        url=f'https://huggingface.co/xiaoyao9184/easyocr/resolve/{revision}/{name}.pth'
        request=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            assert response.status==206
            size=int(response.headers['Content-Range'].split('/')[-1])
            response.read()
        cache=OUT/'models'/('parts_'+name);cache.mkdir(exist_ok=True)
        chunk=2**22
        def fetch(i):
            a=i*chunk;b=min(size,(i+1)*chunk)-1
            part=cache/str(i)
            if part.exists() and part.stat().st_size==b-a+1:return i
            for attempt in range(3):
                try:
                    req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
                    with urllib.request.urlopen(req,timeout=120) as r:
                        assert r.status==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{size}'
                        with part.open('wb') as f:
                            while block:=r.read(2**18):f.write(block)
                    assert part.stat().st_size==b-a+1
                    return i
                except Exception:
                    if attempt==2:raise
        with ThreadPoolExecutor(8) as pool:
            for future in as_completed([pool.submit(fetch,i) for i in range((size+chunk-1)//chunk)]):
                print(name,'part_complete',future.result(),flush=True)
        final=OUT/'models'/(name+'.pth')
        with final.open('wb') as f:
            for i in range((size+chunk-1)//chunk):f.write((cache/str(i)).read_bytes())
        with final.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
        assert actual==md5,(name,actual)
        with final.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        for p in cache.iterdir():p.unlink()
        cache.rmdir()
        records.append(dict(filename=final.name,url=url,size=size,official_md5=md5,verified_md5=actual,sha256=sha))
        print(name,'OFFICIAL_CHECKSUM_VERIFIED',flush=True)
    (OUT/'model_sources.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
