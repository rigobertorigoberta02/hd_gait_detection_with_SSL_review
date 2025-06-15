import os
from pathlib import Path


def mount_drive(mount_point="/content/drive"):
    """Mount Google Drive at the given mount point."""
    from google.colab import drive
    drive.mount(mount_point)


def create_directories(base_dir):
    """Create required subdirectories under base_dir."""
    subdirs = [
        "raw",
        "processed",
        os.path.join("model_outputs"),
        os.path.join("model_outputs", "figs"),
    ]
    for sub in subdirs:
        Path(base_dir, sub).mkdir(parents=True, exist_ok=True)


def export_env_vars(base_dir):
    """Set environment variables for the project."""
    env = {
        "RAW_DATA_AND_LABELS_DIR": os.path.join(base_dir, "raw"),
        "PROCESSED_DATA_DIR": os.path.join(base_dir, "processed"),
        "OUTPUT_DIR": os.path.join(base_dir, "model_outputs"),
        "VIZUALIZE_DIR": os.path.join(base_dir, "model_outputs", "figs"),
        "PACE_DAILY_DATA_DIR": os.path.join(base_dir, "pace_daily", "data"),
        "PACE_DAILY_TARGET_DIR": os.path.join(base_dir, "pace_daily", "labels"),
    }
    for key, val in env.items():
        os.environ[key] = val
    return env


def print_summary(base_dir, env):
    """Display a summary of the created directories and variables."""
    print("Setup summary:\n")
    print(f"BASE_DIR: {base_dir}")
    for key, val in env.items():
        print(f"{key}: {val}")


def main():
    mount_drive("/content/drive")
    base_dir = "/content/drive/MyDrive/hd_data"
    create_directories(base_dir)
    env = export_env_vars(base_dir)
    print_summary(base_dir, env)


if __name__ == "__main__":
    main()
