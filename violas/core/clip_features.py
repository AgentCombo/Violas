"""CLIP feature extraction and key prediction helpers."""

import os
import re
from typing import List, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
tokenizer = clip.tokenize


def extract_feature(img_path):
    """Extract the CLIP feature vector for a single image."""
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
        feature = feature.cpu().numpy().flatten()
    return feature


def get_key_and_type(folder_name):
    """Extract a semantic key and optional type suffix from a folder name."""
    match = re.match(r"([A-Za-z]+)[_ ]?(.*)", folder_name)
    if match:
        key = match.group(1)
        type_part = match.group(2) if match.group(2) else "NULL"
        return key, type_part
    return folder_name, "NULL"


def extract_all_features_from_folders(root_path):
    """
    Extract image features from every class folder under a dataset root.

    Args:
        root_path: Dataset root directory.

    Returns:
        all_vectors: Mapping from folder name to image vectors.
        all_descriptions: Mapping from folder name to image metadata.
    """
    folders = [
        f
        for f in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, f)) and f != "BACKGROUND_Google"
    ]
    folders.sort()
    print(f"Found {len(folders)} class folders", flush=True)

    all_vectors = {}
    all_descriptions = {}
    global_idx = 0

    print(f"Extracting image features from {root_path}...", flush=True)
    for folder in tqdm(folders):
        folder_path = os.path.join(root_path, folder)
        get_key_and_type(folder)

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
                print(f"Failed to extract features for {img_path}: {e}", flush=True)
                continue

        if vectors:
            all_vectors[folder] = vectors
            all_descriptions[folder] = descriptions

    return all_vectors, all_descriptions


class KeyPredictor:
    """
    Predict semantic keys for image features with CLIP text prompts.

    Keys are encoded once with the CLIP text encoder. During prediction, image
    features are compared against these pre-encoded key features by cosine
    similarity.
    """

    def __init__(self, keys: List[str], prompt_template: str = "a photo of a {}"):
        """
        Args:
            keys: Candidate key list, for example ["airplane", "car", "bird"].
            prompt_template: Text prompt template where {} is replaced by a key.
        """
        self.keys = keys
        self.prompt_template = prompt_template
        self.device = device
        self.model = model
        self._encode_keys()

    @classmethod
    def from_vectormap(cls, vectormap, prompt_template: str = "a photo of a {}"):
        """Create a predictor from all keys stored in a VectorMap."""
        keys = sorted(list(vectormap.data.keys()))
        return cls(keys, prompt_template)

    @classmethod
    def from_folder_names(cls, root_path: str, prompt_template: str = "a photo of a {}"):
        """Create a predictor from dataset folder names."""
        folders = [
            f
            for f in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, f)) and f != "BACKGROUND_Google"
        ]

        keys = sorted(set(folders))
        return cls(keys, prompt_template)

    @classmethod
    def from_key_lists(cls, keys: List[str], prompt_template: str = "a photo of a {}"):
        """Create a predictor from an explicit key list."""
        keys = sorted(set(keys))
        return cls(keys, prompt_template)

    def _encode_keys(self):
        """Pre-encode text features for all candidate keys."""
        texts = [self.prompt_template.format(key) for key in self.keys]
        text_tokens = tokenizer(texts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            self.text_features = self.text_features.cpu().numpy()

        print(f"Pre-encoded text features for {len(self.keys)} keys", flush=True)

    def predict(self, img_path: str, top_k: int = 1) -> Tuple[str, float]:
        """
        Predict keys for one image path.

        Returns:
            If top_k is 1, returns (key, confidence_score). Otherwise returns a
            list of (key, confidence_score) pairs.
        """
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu()

        similarities = (image_features @ self.text_features.T).squeeze(0)
        similarities = similarities.numpy()

        if top_k == 1:
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def predict_from_vector(self, vector: np.ndarray, top_k: int = 1) -> Tuple[str, float]:
        """
        Predict keys directly from an encoded image feature vector.

        Returns:
            If top_k is 1, returns (key, confidence_score). Otherwise returns a
            list of (key, confidence_score) pairs.
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_norm = vector / np.linalg.norm(vector)
        similarities = vector_norm @ self.text_features.T

        if top_k == 1:
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def predict_batch_from_vectors(self, vectors: List[np.ndarray], top_k: int = 1) -> List:
        """Predict keys for a batch of already-encoded vectors."""
        results = []
        for vector in vectors:
            try:
                result = self.predict_from_vector(vector, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Vector prediction failed: {e}", flush=True)
                results.append(None)
        return results

    def get_key_vector(self, key: str) -> np.ndarray:
        """Return the text feature vector for a key, or None if it is missing."""
        if key in self.keys:
            idx = self.keys.index(key)
            return self.text_features[idx]
        return None

    def get_predicted_key_vector(self, vector: np.ndarray) -> np.ndarray:
        """Predict a vector's key and return the corresponding key feature."""
        predicted_key, _ = self.predict_from_vector(vector, top_k=1)
        return self.get_key_vector(predicted_key)

    def get_all_keys(self) -> List[str]:
        """Return all available keys."""
        return self.keys.copy()


if __name__ == "__main__":
    root_path = "dataset/102_Flower_processed"
    predictor = KeyPredictor.from_folder_names(root_path)
    print(len(predictor.get_all_keys()))
    print(predictor.get_all_keys())
