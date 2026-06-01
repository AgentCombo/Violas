"""
Ohsumed ，（C01, C02, ...），
"""

import re
import os
import random
from tqdm import tqdm


def preprocess_ohsumed_text(text: str) -> str:
    """Ohsumed ，"""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    return ' '.join(lines).strip()


def collect_text_data_from_folders(root_path: str, max_files_per_folder: int = None,
                                   sample_ratio: float = None, random_state: int = None):
    """
    ohsumed-all ， C01, C02 ， : root_path: max_files_per_folder: sample_ratio: (0~1)
        random_state: : all_data: {folder_name: [(file_name, processed_text), ...]}
    """
    all_data = {}
    folders = [f for f in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, f)) and f.startswith('C') and len(f) == 3]
    folders.sort()

    if random_state is not None:
        rng = random.Random(random_state)
    else:
        rng = random

    print(f"Found {len(folders)} folders")
    if sample_ratio is not None:
        print(f"Sampling {sample_ratio*100:.1f}% of files (random_state={random_state})")

    for folder in tqdm(folders, desc="Encoding batches"):
        folder_path = os.path.join(root_path, folder)
        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))]
        files.sort()

        if sample_ratio is not None and 0 < sample_ratio < 1 and len(files) > 0:
            n_sample = max(1, int(len(files) * sample_ratio))
            files = rng.sample(files, min(n_sample, len(files)))
        elif max_files_per_folder is not None and max_files_per_folder > 0:
            files = files[:max_files_per_folder]

        folder_data = []
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    raw_text = f.read()
                text = preprocess_ohsumed_text(raw_text)
                if text.strip():
                    folder_data.append((file_name, text))
            except Exception:
                continue

        if folder_data:
            all_data[folder] = folder_data

    print(f"\nCollected in total {len(all_data)} folder data")
    return all_data
