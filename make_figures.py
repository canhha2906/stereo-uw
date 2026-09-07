"""Qualitative figures for the paper. The one compute task the paper still needs.

Row per scene: left image | ground truth | direct-transfer prediction | ours (turbid4).
Colour map and disparity scale are shared across a row so the panels are comparable.
"""
import os, re
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def predict(m, lp, rp):
    L,R=load(lp).unsqueeze(0).cuda(),load(rp).unsqueeze(0).cuda()
    ph=(32-L.shape[2]%32)%32
    if ph: L=torch.nn.functional.pad(L,(0,0,ph,0)); R=torch.nn.functional.pad(R,(0,0,ph,0))
    with torch.no_grad(): o=m(L,R)
    p=(o[0] if isinstance(o,(list,tuple)) else o).squeeze(0)
    return (p[ph:,:] if ph else p).cpu().numpy()

def build(ckpt):
    m=Fast_ACVNet_plus(192,False).cuda().eval()
    c=torch.load(ckpt,map_location="cpu",weights_only=False)
    m.load_state_dict({k.replace("module.",""):v for k,v in c["model"].items()})
    return m

items=index(ROOT)
perm=np.random.default_rng(0).permutation(len(items))
test=[items[i] for i in perm[int(0.8*len(items))+int(0.1*len(items)):]]

# one representative frame per scene, taken from the test split only
picks={}
for lp,rp,dp,sc in test:
    if sc not in picks: picks[sc]=(lp,rp,dp)
    if len(picks)==4: break

base=build("./pretrained/Fast-ACVNet+/kitti_2015.ckpt")
ours=build("./checkpoints/exp_turbid4/checkpoint_000059.ckpt")

order=["coral","ship","industry","default"]
order=[s for s in order if s in picks]
fig,axes=plt.subplots(len(order),4,figsize=(15,2.6*len(order)))
if len(order)==1: axes=axes[None,:]
cols=["Left image","Ground truth","Direct transfer","Ours (physics-rendered)"]

for r,sc in enumerate(order):
    lp,rp,dp=picks[sc]
    rgb=np.asarray(Image.open(lp).convert("RGB"))
    gt=read_pfm(dp).astype(np.float32)
    pb=predict(base,lp,rp); po=predict(ours,lp,rp)
    h=min(gt.shape[0],pb.shape[0]); w=min(gt.shape[1],pb.shape[1])
    gt,pb,po=gt[:h,:w],pb[:h,:w],po[:h,:w]
    valid=(gt>0)&(gt<192)&np.isfinite(gt)
    vmax=float(np.percentile(gt[valid],99)) if valid.any() else 192.
    gtv=np.where(valid,gt,np.nan)
    e_b=np.abs(pb[valid]-gt[valid]).mean(); e_o=np.abs(po[valid]-gt[valid]).mean()
    for c,(img,ttl) in enumerate(zip([rgb,gtv,pb,po],cols)):
        ax=axes[r,c]
        if c==0: ax.imshow(img)
        else: ax.imshow(img,cmap="magma",vmin=0,vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(ttl,fontsize=11)
        if c==0: ax.set_ylabel(sc,fontsize=11)
        if c==2: ax.set_xlabel(f"EPE {e_b:.2f}",fontsize=9)
        if c==3: ax.set_xlabel(f"EPE {e_o:.2f}",fontsize=9)
    print(f"{sc:10s} direct {e_b:6.3f}   ours {e_o:6.3f}", flush=True)

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
for ext in ("png","pdf"):
    plt.savefig(f"figures/qualitative.{ext}", dpi=200, bbox_inches="tight")
print("wrote figures/qualitative.png and .pdf")
