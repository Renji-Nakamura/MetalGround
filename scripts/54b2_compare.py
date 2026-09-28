#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('results/0054b_reference');M=Path('results/metalground_metal_preprocess_0054b2.json');O=Path('results/metalground_preprocess_compare_0054b2.json')
def st(x):
 x=np.asarray(x,np.float64);a=np.abs(x);return {'mean_abs':float(a.mean()),'median_abs':float(np.median(a)),'p95_abs':float(np.percentile(a,95)),'p99_abs':float(np.percentile(a,99)),'max_abs':float(a.max()),'rmse':float(np.sqrt(np.mean(x*x)))}
meta=json.loads((R/'metadata.json').read_text());m=json.loads(M.read_text());shape=tuple(meta['processor']['output_shape_chw']);ref=np.fromfile(R/meta['reference_file'],np.float32).reshape(shape);cand=np.fromfile(m['output_file'],np.float32).reshape(shape);d=st(cand-ref)
pil=np.asarray(Image.open(R/meta['letterbox']['reference_png']).convert('RGB'),np.int16);let=np.fromfile(m['diagnostic_letterbox_file'],np.uint8).reshape(901,1200,4)[...,:3].astype(np.int16);ld=st(let-pil);exact=float(np.mean(np.all(let==pil,axis=-1)))
g={'shape_exact':list(cand.shape)==list(ref.shape),'mean_abs_le_0_010':d['mean_abs']<=.010,'p99_abs_le_0_050':d['p99_abs']<=.050,'max_abs_le_0_250':d['max_abs']<=.250,'metal_p95_ms_le_3':float(m['timing_ms']['p95'])<=3.0}
r={'experiment':'0054b-2','difference':d,'pass1_letterbox_difference_u8':ld,'pass1_exact_rgb_triplet_fraction':exact,'metal_timing_ms':m['timing_ms'],'original_0054b_preregistered_gates_reapplied_unchanged':g,'all_original_gates_pass':all(g.values())};O.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));print('Saved:',O)
