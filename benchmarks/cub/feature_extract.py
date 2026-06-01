import torch
import re
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
from typing import Any, List, Tuple

# Load feature extraction model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
tokenizer = clip.tokenize


def extract_feature(img_path):
    """Extract the feature vector of a single image"""
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
        feature = feature.cpu().numpy().flatten()
    return feature


def get_key_and_type(folder_name):
    """Extract key and type from CUB folder name. CUB format: 001.Black_footed_Albatross"""
    m = re.match(r"^\d+\.(.+)$", folder_name)
    if m:
        key = m.group(1)  # Black_footed_Albatross
        return key, "NULL"
    return folder_name, "NULL"


def extract_all_features_from_folders(root_path):
    """
    Extract image features from all category folders under the images root directory of the CUB dataset.
    The root_path should be .../CUB_200_2011/images, with subfolders such as 001.Black_footed_Albatross under it.

    Parameters:
        root_path: images root directory path

    Return:
        all_vectors: {folder_name: [vector1, vector2, ...]}
        all_descriptions: {folder_name: [{'img_path': ..., 'idx': ...}, ...]}
    """
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    # folders = folders[:3] # Small sample run-through test
    folders.sort()

    all_vectors = {}
    all_descriptions = {}
    global_idx = 0

    print(f"from path {root_path} Extract all image features...", flush=True)
    for folder in tqdm(folders):
        folder_path = os.path.join(root_path, folder)
        key, type_part = get_key_and_type(folder)

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        images.sort()

        vectors = []
        descriptions = []
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                vec = extract_feature(img_path)
                vectors.append(vec)
                descriptions.append({"img_path": img_path, "idx": global_idx})
                global_idx += 1
            except Exception as e:
                print(f"picture {img_path} Feature extraction failed: {e}", flush=True)
                continue

        if vectors:
            all_vectors[folder] = vectors
            all_descriptions[folder] = descriptions

    return all_vectors, all_descriptions


class KeyPredictor:
    """
    Use the CLIP model to predict the key (category) of the image.
    Same implementation as caltech-bench, supports creation from key list.
    """

    def __init__(self, keys: List[str], prompt_template: str = "a photo of a {}"):
        self.keys = keys
        self.prompt_template = prompt_template
        self.device = device
        self.model = model
        self._encode_keys()

    @classmethod
    def from_vectormap(cls, vectormap, prompt_template: str = "a photo of a {}"):
        keys = sorted(list(vectormap.data.keys()))
        return cls(keys, prompt_template)

    @classmethod
    def from_folder_names(cls, root_path: str, prompt_template: str = "a photo of a {}"):
        folders = [
            f
            for f in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, f))
        ]
        keys = sorted(set(folders))
        return cls(keys, prompt_template)

    @classmethod
    def from_key_lists(cls, keys: List[str], prompt_template: str = "a photo of a {}"):
        keys = sorted(set(keys))
        return cls(keys, prompt_template)

    def _encode_keys(self):
        texts = [self.prompt_template.format(key) for key in self.keys]
        text_tokens = tokenizer(texts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            self.text_features = self.text_features.cpu().numpy()
        print(f"precoded {len(self.keys)} text features of keys", flush=True)

    def predict(self, img_path: str, top_k: int = 1) -> Tuple[str, float]:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu()
        similarities = (image_features @ self.text_features.T).squeeze(0).numpy()
        if top_k == 1:
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def predict_from_vector(self, vector: np.ndarray, top_k: int = 1) -> Tuple[str, float]:
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        vector_norm = vector / np.linalg.norm(vector)
        similarities = vector_norm @ self.text_features.T
        if top_k == 1:
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def get_key_vector(self, key: str) -> np.ndarray:
        if key in self.keys:
            return self.text_features[self.keys.index(key)]
        return None

    def get_predicted_key_vector(self, vector: np.ndarray) -> np.ndarray:
        predicted_key, _ = self.predict_from_vector(vector, top_k=1)
        return self.get_key_vector(predicted_key)

    def get_all_keys(self) -> List[str]:
        return self.keys.copy()


if __name__ == "__main__":
    root_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset", "CUB_200_2011", "CUB_200_2011", "images"
    )
    predictor = KeyPredictor.from_folder_names(root_path)
    print(len(predictor.get_all_keys()))
    print(predictor.get_all_keys()[:5])
