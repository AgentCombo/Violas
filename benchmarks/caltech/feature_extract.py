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
    """Extract key and type from folder name"""
    m = re.match(r"([A-Za-z]+)[_ ]?(.*)", folder_name)
    if m:
        key = m.group(1)
        type_part = m.group(2) if m.group(2) else "NULL"
        return key, type_part
    return folder_name, "NULL"


def extract_all_features_from_folders(root_path):
    """
    Extract image features from all folders in the specified root directory

    Parameters:
        root_path: root directory path

    Return:
        all_vectors: {folder_name: [vector1, vector2, ...]}
        all_descriptions: {folder_name: [{'img_name': img_name1, 'idx': idx1}, ...]}
    """
    # Get all folders, filter out BACKGROUND_Google
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and f != "BACKGROUND_Google"]
    folders.sort()


    # Vector to store all images
    all_vectors = {}
    all_descriptions = {}

    # Global index counter
    global_idx = 0

    print(f"from path {root_path} Extract all image features...", flush=True)
    for folder in tqdm(folders):
        folder_path = os.path.join(root_path, folder)
        key, type_part = get_key_and_type(folder)

        images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        images.sort()

        vectors = []
        descriptions = []
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                vec = extract_feature(img_path)
                vectors.append(vec)
                descriptions.append({'img_path': img_path, 'idx': global_idx})
                global_idx += 1
            except Exception as e:
                print(f"picture{img_path}Feature extraction failed: {e}", flush=True)
                continue

        if vectors:
            all_vectors[folder] = vectors
            all_descriptions[folder] = descriptions

    return all_vectors, all_descriptions


class KeyPredictor:
    """
    Use the CLIP model to predict the key (category) of the image

    How it works:
        1. During initialization (_encode_keys):
           - Convert each key to text via prompt_template (e.g. "airplane" -> "a photo of a airplane"）
           - Encode all text into text feature vectors using the CLIP text encoder
           - Precoding results are stored in self.text_features

        2. When predicting:
           - Use the CLIP image encoder to encode the input image into an image feature vector
           - Calculate cosine similarity between image features and all precoded text features
           - Return the key with the highest similarity as the prediction result

    How to use:
        # Initialized from VectorMap
        predictor = KeyPredictor.from_vectormap(vectormap)

        # Or manually specify the key list
        predictor = KeyPredictor(keys=['airplane', 'car', 'bird', ...])

        # key key, confidence = predictor.predict(img_path)

        # Or predict multiple pictures
        results = predictor.predict_batch([img_path1, img_path2, ...])
    """

    def __init__(self, keys: List[str], prompt_template: str = "a photo of a {}"):
        """
        KeyPredictor : keys: key ， ['airplane', 'car', 'bird', ...]
            prompt_template: text prompt template,{} will be replaced by key
        """
        self.keys = keys
        self.prompt_template = prompt_template
        self.device = device
        self.model = model

        # Pre-encode text features for all keys
        self._encode_keys()

    @classmethod
    def from_vectormap(cls, vectormap, prompt_template: str = "a photo of a {}"):
        """
        Extract all keys from VectorMap and create KeyPredictor

        Parameters:
            vectormap: VectorMap object
            prompt_template: text prompt template
        """
        keys = sorted(list(vectormap.data.keys()))
        return cls(keys, prompt_template)

    @classmethod
    def from_folder_names(cls, root_path: str, prompt_template: str = "a photo of a {}"):
        """
        Extract key from folder name and create KeyPredictor

        Parameters:
            root_path: Data set root directory
            prompt_template: text prompt template
        """
        folders = [f for f in os.listdir(root_path)
                  if os.path.isdir(os.path.join(root_path, f)) and f != "BACKGROUND_Google"]

        keys = sorted(set(folders))
        return cls(keys, prompt_template)

    @classmethod
    def from_key_lists(cls, keys: List[str], prompt_template: str = "a photo of a {}"):
        """
        key KeyPredictor : keys: key ， ['airplane', 'car', 'bird', ...]
            prompt_template: Text prompt template
        """
        keys = sorted(set(keys))
        return cls(keys, prompt_template)

    def _encode_keys(self):
        """Pre-encode text features for all keys

        Prompt_template is used here to convert key into a text prompt, for example:
        - key = "airplane", prompt_template = "a photo of a {}"
          -> generate text "a photo of a airplane"
        - key = "car", prompt_template = "an image of {}"
          -> generate text "an image of car"

        These text hints are CLIP-encoded into text features for subsequent similarity calculations.
        """
        # Build text prompts: use prompt_template to convert each key into a complete text description
        # For example: key="airplane" + template="a photo of a {}" -> "a photo of a airplane"
        texts = [self.prompt_template.format(key) for key in self.keys]

        # Using CLIP’s tokenizer and text encoder
        text_tokens = tokenizer(texts).to(self.device)

        with torch.no_grad():
            # Encode text features
            self.text_features = self.model.encode_text(text_tokens)
            # Normalized features (used for cosine similarity calculation)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            # Convert to numpy array for subsequent use
            self.text_features = self.text_features.cpu().numpy()

        print(f"precoded {len(self.keys)} text features of keys", flush=True)

    def predict(self, img_path: str, top_k: int = 1) -> Tuple[str, float]:
        """
        Predict the key of a single image

        Workflow:
        1. Encoding pictures: Encoding input pictures into image feature vectors
        2. Calculate similarity: calculate cosine similarity with precoded key text features (self.text_features)
        3. Return results: select the key with the highest similarity as the prediction result

        Parameters:
            img_path: image path
            top_k: Returns the top k most likely keys (default returns the most likely 1)

        Return:
            if top_k=1: (key, confidence_score)
            if top_k>1: [(key1, score1), (key2, score2), ...]

        ： - self.text_features _encode_keys() - ：key -> prompt_template -> text -> CLIPText encoding -> text features
            - Prediction Process: Pictures -> CLIPImage encoding -> Image features -> Compare with text features -> most similar key
        """
        # Step 1: Load and preprocess images
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Step 2: Encode image features (using CLIP’s image encoder)
            image_features = self.model.encode_image(image)
            # Normalized features (used for cosine similarity calculation)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu()

        # Step 3: Calculate the similarity (cosine similarity) with all pre-encoded key text features
        # self.text_features is precoded during initialization, and each key corresponds to a text feature vector
        similarities = (image_features @ self.text_features.T).squeeze(0)
        similarities = similarities.numpy()

        if top_k == 1:
            # Return the most similar key
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])
        else:
            # Returns the top k most similar keys
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def predict_from_vector(self, vector: np.ndarray, top_k: int = 1) -> Tuple[str, float]:
        """
        Predict key directly from vector (no image path required)

        Parameters:
            vector: encoded image feature vector (numpy array)
            top_k: Returns the top k most likely keys (default returns the most likely 1)

        Return:
            if top_k=1: (key, confidence_score)
            if top_k>1: [(key1, score1), (key2, score2), ...]
        """
        # Make sure the vector is a numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        # Normalized vector (used for cosine similarity calculation)
        vector_norm = vector / np.linalg.norm(vector)

        # text_features is already a numpy array (converted to CPU numpy in _encode_keys)
        # Calculate cosine similarity directly
        similarities = vector_norm @ self.text_features.T

        if top_k == 1:
            # Return the most similar key
            best_idx = np.argmax(similarities)
            return self.keys[best_idx], float(similarities[best_idx])
        else:
            # Returns the top k most similar keys
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.keys[idx], float(similarities[idx])) for idx in top_indices]

    def predict_batch_from_vectors(self, vectors: List[np.ndarray], top_k: int = 1) -> List:
        """
        key : vectors: top_k: k key : ， predict_from_vector()
        """
        results = []
        for vector in vectors:
            try:
                result = self.predict_from_vector(vector, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Prediction vector failed: {e}", flush=True)
                results.append(None)
        return results

    def get_key_vector(self, key: str) -> np.ndarray:
        """
        Get the key vector of the specified key (text_features)

        Parameters:
            key: key name

        Return:
            key vector (numpy array), returns None if key does not exist
        """
        if key in self.keys:
            idx = self.keys.index(key)
            return self.text_features[idx]
        return None

    def get_predicted_key_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Predict the key based on the query vector and return the key vector corresponding to the key

        Parameters:
            vector: Query image feature vector

        Return:
            The key vector corresponding to the predicted key (text_features)
        """
        predicted_key, _ = self.predict_from_vector(vector, top_k=1)
        return self.get_key_vector(predicted_key)

    def get_all_keys(self) -> List[str]:
        """Get a list of all available keys"""
        return self.keys.copy()


if __name__ == "__main__":
    root_path = "dataset/caltech-101/101_ObjectCategories"
    predictor = KeyPredictor.from_folder_names(root_path)
    print(len(predictor.get_all_keys()))
    print(predictor.get_all_keys())
