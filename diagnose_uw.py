"""Where does the Stage-4 model actually fail on UWStereo? Break error down by scene
and by ground-truth disparity range, so the next step is chosen from evidence."""
import os, re, argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
from models.Fast_ACV_plus import Fast_ACVNet_plus

ROOT = r"C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo"
SCENES = [("default","default"),("coral reef","coral"),("industry","industry"),("ship split","ship")]
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

def read_pfm(p):
    with open(p,"rb") as f:
        f.readline()
        while True:
            l=f.readline().decode("latin-1")
            if not l.startswith("#"): break
        w,h=map(int,re.match(r"^(\d+)\s+(\d+)\s*$",l).groups()); sc=float(f.readline().rstrip())
        return np.flipud(np.fromfile(f,"<f" if sc<0 else ">f").reshape(h,w))

def index(root):
    out=[]
    for outer,inner in SCENES:
        b=Path(root)/outer/inner; dd=b/"disparity"
        if not dd.is_dir(): continue
        for d in sorted(os.listdir(dd)):
            if d.endswith(".pfm"):
                s=d[:-4]
                out.append((str(b/"images"/"left"/f"{s}.png"), str(b/"images"/"right"/f"{s}.png"), str(dd/d), inner))
    return out

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),np.float32)/255.
    return torch.from_numpy(((a-MEAN)/STD).transpose(2,0,1))

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", default="./checkpoints/kitti_uw_III/checkpoint_000053.ckpt")
ap.add_argument("--limit", type=int, default=400)
a=ap.parse_args()

items=index(ROOT)
perm=np.random.default_rng(0).permutation(len(items))
test=[items[i] for i in perm[int(0.8*len(items))+int(0.1*len(items)):]][:a.limit]

m=Fast_ACVNet_plus(192,False).cuda().eval()
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
m.load_state_dict({k.replace("module.",""):v for k,v in ck["model"].items()})

by_scene={}; buckets={"0-32":[], "32-64":[], "64-128":[], "128-192":[]}
frac_over=[]
for lp,rp,dp,scene in test:
    L,R=load(lp).unsqueeze(0).cuda(),load(rp).unsqueeze(0).cuda()
    g=torch.from_numpy(np.ascontiguousarray(read_pfm(dp))).float().cuda()
    ph=(32-L.shape[2]%32)%32
    if ph: L=torch.nn.functional.pad(L,(0,0,ph,0)); R=torch.nn.functional.pad(R,(0,0,ph,0))
    with torch.no_grad(): o=m(L,R)
    p=(o[0] if isinstance(o,(list,tuple)) else o).squeeze(0)
    if ph: p=p[ph:,:]
    finite=torch.isfinite(g)&(g>0)
    frac_over.append(((g>=192)&finite).float().sum().item()/max(finite.float().sum().item(),1)*100)
    mask=finite&(g<192)
    if mask.sum()==0: continue
    e=(p[mask]-g[mask]).abs()
    by_scene.setdefault(scene,[]).append(e.mean().item())
    gv=g[mask]
    for name,(lo,hi) in {"0-32":(0,32),"32-64":(32,64),"64-128":(64,128),"128-192":(128,192)}.items():
        sel=(gv>=lo)&(gv<hi)
        if sel.sum()>0: buckets[name].append((p[mask][sel]-gv[sel]).abs().mean().item())

print(f"N = {len(test)}\n")
print("EPE by scene:")
for k,v in sorted(by_scene.items(), key=lambda x:-np.mean(x[1])):
    print(f"  {k:10s} n={len(v):4d}  EPE {np.mean(v):7.4f}")
print("\nEPE by ground-truth disparity range:")
for k,v in buckets.items():
    if v: print(f"  {k:9s} n={len(v):4d}  EPE {np.mean(v):7.4f}")
print(f"\nGT pixels beyond the model's 192 range: {np.mean(frac_over):.2f}% (excluded from EPE)")
