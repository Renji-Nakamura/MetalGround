#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
PRECISION_BITS=22

def filt(x):
    x=abs(x)
    return 1.0-x if x<1.0 else 0.0

def coeffs(in_size,out_size):
    scale=in_size/out_size
    fs=max(scale,1.0); support=fs
    ksize=math.ceil(support)*2+1
    b=np.zeros((out_size,2),np.int32)
    c=np.zeros((out_size,ksize),np.int32)
    inv=1.0/fs
    for xx in range(out_size):
        center=(xx+0.5)*scale
        xmin=max(int(center-support+0.5),0)
        xmax=min(int(center+support+0.5),in_size)
        n=xmax-xmin
        ws=[filt((x+xmin-center+0.5)*inv) for x in range(n)]
        s=sum(ws)
        if s: ws=[w/s for w in ws]
        qs=[]
        for w in ws:
            qs.append(int((-0.5 if w<0 else 0.5)+w*(1<<PRECISION_BITS)))
        b[xx]=[xmin,n]
        c[xx,:n]=np.asarray(qs,np.int32)
    return b,c,ksize

def save(out,name,i,o):
    b,c,k=coeffs(i,o)
    bf=f'{name}_bounds_i32.bin'; cf=f'{name}_coeffs_i32.bin'
    b.tofile(out/bf); c.tofile(out/cf)
    return {'in_size':i,'out_size':o,'ksize':k,'bounds_file':bf,'coeffs_file':cf,'precision_bits':PRECISION_BITS}

def main():
    out=Path('results/0054b_pillow_coeffs'); out.mkdir(parents=True,exist_ok=True)
    m={'experiment':'0054b-2','precision_bits':PRECISION_BITS,'stages':{
      'stage1_h':save(out,'stage1_h',1280,1200),
      'stage1_v':save(out,'stage1_v',720,675),
      'stage2_h':save(out,'stage2_h',1200,1065),
      'stage2_v':save(out,'stage2_v',901,800)}}
    (out/'metadata.json').write_text(json.dumps(m,indent=2))
    print('=== 0054b-2 Pillow coefficient tables ===')
    for n,s in m['stages'].items(): print(n,'ksize',s['ksize'])
if __name__=='__main__': main()
