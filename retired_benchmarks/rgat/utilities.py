import os
import time

def create_ckpt_folder(base_dir, prefix="ckpt"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"{prefix}_{timestamp}" if prefix else timestamp
    full_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

