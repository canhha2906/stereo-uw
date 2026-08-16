"""Smoke-test the UWStereo dataloader on the actual files."""
from data import UWStereoDataset

ROOT = r'C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo'


def main():
    ds = UWStereoDataset(ROOT, split='train', d_max=256, augment=True)
    print(f'train size: {len(ds)}')
    s = ds[0]
    print({k: tuple(v.shape) for k, v in s.items()})
    print(f'left dtype={s["left"].dtype} range=[{s["left"].min():.2f},{s["left"].max():.2f}]')
    print(f'disp dtype={s["disp"].dtype} range=[{s["disp"].min():.2f},{s["disp"].max():.2f}]')
    print(f'valid frac={s["valid"].mean():.3f}')

    val_ds = UWStereoDataset(ROOT, split='val', d_max=256, augment=False)
    test_ds = UWStereoDataset(ROOT, split='test', d_max=256, augment=False)
    total = len(ds) + len(val_ds) + len(test_ds)
    print(f'val={len(val_ds)} test={len(test_ds)} total={total}')


if __name__ == '__main__':
    main()
