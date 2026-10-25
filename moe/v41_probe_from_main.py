#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
inputs=[Path(x) for x in sys.argv[1].split(',')]
out=Path(sys.argv[2]); seq=int(sys.argv[3]) if len(sys.argv)>3 else 256
out.mkdir(parents=True, exist_ok=True); (out/'h').mkdir(exist_ok=True)
rs=8+3*5120*4
ids_list=[]; arg_list=[]; metas=[]; w=0
for path in inputs:
    raw=np.memmap(path,dtype=np.uint8,mode='r'); n=raw.size//rs
    raw=raw[:n*rs].reshape(n,rs)
    tok=raw[:,:4].copy().view('<i4')[:,0].astype(np.int64)
    pos=raw[:,4:8].copy().view('<i4')[:,0]
    taps=raw[:,8:].copy().view('<f4').reshape(n,3,5120).astype(np.float32).mean(axis=1)
    nw=(n-1)//seq
    for k in range(nw):
        s=k*seq
        ids_list.append(tok[s:s+seq]); arg_list.append(tok[s+1:s+seq+1])
        taps[s:s+seq].astype(np.float32).tofile(out/'h'/f'win_{w:02d}_h.bin')
        tok[s+1:s+seq+1].astype(np.int32).tofile(out/'h'/f'win_{w:02d}_argmax.bin')
        w+=1
    metas.append({'path':str(path),'records':int(n),'windows':int(nw),'first_pos':int(pos[0]),'last_pos':int(pos[-1])})
np.stack(ids_list).astype(np.int32).tofile(out/'tokens.bin')
meta={'sources':metas,'seq':seq,'windows':w,'hidden':5120,'taps':'mean of 3 x 5120'}
(out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
