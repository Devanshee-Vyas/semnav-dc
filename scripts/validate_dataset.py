"""Validate SS4Blind dataset paths before training."""
import argparse
from pathlib import Path
import pandas as pd
import sys

# Import resolve_path from the dataset module
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.ss4blind import resolve_path


def validate_split(csv_path: Path, root: Path, split_name: str):
    """Validate all paths in a split CSV."""
    print(f"\n{'='*60}")
    print(f"Validating {split_name} split: {csv_path}")
    print('='*60)
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    total = len(df)
    resolved = 0
    missing = []
    
    print(f"Total rows: {total}")
    
    for idx, row in df.iterrows():
        try:
            # Try to resolve RGB
            rgb_path = resolve_path(row['rgb_path'], root, [".png", ".jpg", ".jpeg"])
            
            # Try to resolve depth
            depth_path = resolve_path(row['depth_path'], root, [".npy", ".png"])
            
            # Try semantic if present (optional)
            if 'sem_path' in row and pd.notna(row['sem_path']) and str(row['sem_path']).strip():
                try:
                    sem_path = resolve_path(row['sem_path'], root, [".png"])
                except FileNotFoundError:
                    pass  # Semantic is optional
            
            resolved += 1
        except FileNotFoundError as e:
            if len(missing) < 5:  # Show first 5 errors
                missing.append((idx, str(e)))
    
    print(f"✓ Resolved: {resolved}/{total}")
    
    if missing:
        print(f"\n❌ Missing: {total - resolved}/{total}")
        print("\nFirst errors:")
        for idx, error in missing:
            print(f"  Row {idx}: {error}")
        return False
    else:
        print(f"✅ All paths resolved successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate SS4Blind dataset paths')
    parser.add_argument('--splits_dir', type=str, required=True,
                       help='Directory containing train/val/test CSV files')
    parser.add_argument('--root', type=str, required=True,
                       help='Root directory for resolving relative paths')
    args = parser.parse_args()
    
    splits_dir = Path(args.splits_dir)
    root = Path(args.root)
    
    print("="*60)
    print("SS4Blind Dataset Path Validation")
    print("="*60)
    print(f"Splits directory: {splits_dir}")
    print(f"Data root: {root}")
    
    all_valid = True
    
    for split in ['train', 'val', 'test']:
        csv_path = splits_dir / f'{split}.csv'
        if csv_path.exists():
            valid = validate_split(csv_path, root, split)
            all_valid = all_valid and valid
        else:
            print(f"\n⚠ Skipping {split} (CSV not found)")
    
    print("\n" + "="*60)
    if all_valid:
        print("✅ VALIDATION PASSED")
        print("="*60)
        print("\nReady to train!")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("="*60)
        print("\nFix missing files before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

