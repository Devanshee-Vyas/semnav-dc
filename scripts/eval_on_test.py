# scripts/eval_on_test.py
import torch, numpy as np
from torch.utils.data import DataLoader
from src.data.ss4blind_dataset import SS4BlindDataset
from src.models.unet_baseline import UNetSmall

def _align(p,g,v):
    if p.ndim==3: 
        p=p.unsqueeze(1)
    if g.ndim==3: 
        g=g.unsqueeze(1)
    if v.ndim==3: 
        v=v.unsqueeze(1)
    if v.max()>1.5: 
        v=(v>0).float()
    return p,g,v

def mae(p,g,v):
    p,g,v=_align(p,g,v); 
    d=v.sum().clamp_min(1.0)
    return ((p-g).abs()*v).sum()/d

def rmse(p,g,v):
    p,g,v=_align(p,g,v); d=v.sum().clamp_min(1.0)
    return torch.sqrt(((p-g)**2*v).sum()/d)

root="src/data/ss4blind/processed"
dl=DataLoader(SS4BlindDataset(root,"test",use_sparse=True),batch_size=8,shuffle=False,num_workers=0)
m=UNetSmall(5,1); 
m.load_state_dict(torch.load("checkpoints/ss4blind_unet/best.pth", map_location="cpu")); 
m.eval()
ms,as_=[] ,[]
num = 0
with torch.no_grad():
    for b in dl:
        print(num)
        num += 1
        x=b["x"]; 
        g=b["depth_gt"]; 
        v=b["valid_gt"]
        p=m(x)
        ms.append(rmse(p,g,v).item()); 
        as_.append(mae(p,g,v).item())
print(f"TEST  RMSE={np.mean(ms):.4f}  MAE={np.mean(as_):.4f}")
