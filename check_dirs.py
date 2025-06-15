import os
from pathlib import Path


def check_dirs():
    """Check required environment directories."""
    missing = []
    # Check RAW_DATA_AND_LABELS_DIR
    raw_dir = os.environ.get('RAW_DATA_AND_LABELS_DIR')
    if not raw_dir:
        missing.append("RAW_DATA_AND_LABELS_DIR not set")
    else:
        path = Path(raw_dir)
        if not path.is_dir():
            missing.append(f"{raw_dir} does not exist")
        else:
            has_data = any(path.glob('*.csv')) or any(path.glob('*.npz'))
            if not has_data:
                missing.append(f"{raw_dir} has no .csv or .npz files")

    # Check PROCESSED_DATA_DIR
    processed_dir = os.environ.get('PROCESSED_DATA_DIR')
    if not processed_dir:
        missing.append("PROCESSED_DATA_DIR not set")
    else:
        p_path = Path(processed_dir)
        if not p_path.is_dir():
            missing.append(f"{processed_dir} does not exist")
        elif not any(p_path.iterdir()):
            missing.append(f"{processed_dir} is empty")

    # Check OUTPUT_DIR/checkpoints writability
    output_dir = os.environ.get('OUTPUT_DIR')
    if not output_dir:
        missing.append("OUTPUT_DIR not set")
    else:
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            test_file = checkpoint_dir / '.write_test'
            with open(test_file, 'w'):
                pass
            test_file.unlink()
        except Exception:
            missing.append(f"Cannot write to {checkpoint_dir}")

    if missing:
        for msg in missing:
            print(msg)
    else:
        print("OK")


if __name__ == '__main__':
    check_dirs()
