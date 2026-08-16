"""Smoke-test the combined SceneFlow loader on the actual D: extractions."""
from data import SceneFlowDataset

ROOT = r"D:\SCENEFLOW"


def main():
    print("=== TRAIN split ===")
    tr = SceneFlowDataset(ROOT, split="train", d_max=256, augment=True,
                          crop_h=256, crop_w=512)
    s = tr[0]
    print({k: tuple(v.shape) for k, v in s.items()})
    print(f"left  range=[{s['left'].min():.2f},{s['left'].max():.2f}]")
    print(f"disp  range=[{s['disp'].min():.2f},{s['disp'].max():.2f}]")
    print(f"valid frac={s['valid'].mean():.3f}")

    print("\n=== TEST split ===")
    te = SceneFlowDataset(ROOT, split="test", d_max=256, augment=False)
    print(f"test size: {len(te)}")


if __name__ == "__main__":
    main()
