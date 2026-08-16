"""End-to-end: real UWStereo batch through GwcNet-lite for both agg paths."""
import torch
from torch.utils.data import DataLoader

from data import UWStereoDataset
from models import GwcNetLite
from engine import masked_smooth_l1, metric_epe, metric_d1

ROOT = r'C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo'


def main():
    ds = UWStereoDataset(ROOT, split='train', d_max=256, augment=True, crop_h=128, crop_w=256)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    for agg in ('3d', '2d'):
        m = GwcNetLite(d_max=256, res=8, groups=8, feat_channels=32, agg=agg, upsample='bilinear')
        m.eval()
        with torch.no_grad():
            disp, disp_low = m(batch['left'], batch['right'])
        loss = masked_smooth_l1(disp, batch['disp'], batch['valid']).item()
        epe = metric_epe(disp, batch['disp'], batch['valid']).item()
        d1 = metric_d1(disp, batch['disp'], batch['valid']).item()
        print(f"agg={agg} | disp={tuple(disp.shape)} | loss={loss:.2f} EPE={epe:.2f}px D1={d1*100:.1f}%")
    print("(random-init weights, EPE will be ~mean disparity — that's expected)")


if __name__ == '__main__':
    main()
