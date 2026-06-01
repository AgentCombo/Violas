"""
Ohsumed - extract_all_text_features_from_folders: train/test - KeyPredictor: sentence_transformers key
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List

from text_preprocess import collect_text_data_from_folders
from text_vectorizer import Vectorizer


def extract_all_text_features_from_folders(
    root_path,
    vectorizer_method='tfidf',
    model_name='all-MiniLM-L6-v2',
    test_size=0.1,
    random_state=42,
    sample_ratio=None,
    max_files_per_folder=None,
):
    """
    ohsumed train/test（ TF-IDF，） : train_vectors_by_folder, train_descriptions_by_folder, test_data, vectorizer
    """
    print(f"Removing from {root_path} Read text and preprocess...")
    all_data = collect_text_data_from_folders(
        root_path,
        max_files_per_folder=max_files_per_folder,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )

    print(f"\nInitialize the feature extractor: {vectorizer_method}...")
    vectorizer = Vectorizer(method=vectorizer_method, model_name=model_name)

    all_texts = [item[1] for folder, items in sorted(all_data.items()) for item in items]
    if not all_texts:
        raise ValueError("No text data collected")
    vecs = vectorizer.fit_transform(all_texts)
    dim = vecs.shape[1] if hasattr(vecs, 'shape') else len(all_texts[0])
    print(f"{vectorizer_method} Already in {len(all_texts)} (: {dim})")

    train_vectors_by_folder = {}
    train_descriptions_by_folder = {}
    test_data = {}

    print("\n=== Feature Extraction ===")
    for folder, items in all_data.items():
        if len(items) == 0:
            continue

        file_paths = [f"{folder}/{item[0]}" for item in items]
        texts = [item[1] for item in items]

        vectors = vectorizer.transform(texts)
        vectors = np.array(vectors)

        if len(items) > 1:
            X_train, X_test, path_train, path_test = train_test_split(
                vectors, file_paths, test_size=test_size, random_state=random_state
            )
        else:
            X_train, path_train = vectors, file_paths
            X_test, path_test = [], []

        train_vectors_by_folder[folder] = list(X_train)
        train_descriptions_by_folder[folder] = [{'img_path': path, 'key': folder} for path in path_train]

        test_data[folder] = {
            'vectors': list(X_test),
            'descriptions': [{'img_path': path, 'key': folder} for path in path_test]
        }

    return train_vectors_by_folder, train_descriptions_by_folder, test_data, vectorizer


class KeyPredictor:
    """KeyPredictor， Ohsumed （ TF-IDF sentence_transformers）"""

    def __init__(self, keys: List[str], prompt_template: str = "a medical article about {}", vectorizer=None):
        self.keys = keys
        self.prompt_template = prompt_template
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = Vectorizer(method='sentence_transformers', model_name='all-MiniLM-L6-v2')
            self.vectorizer.fit_transform(["warm up"])
        self._encode_keys()

    @classmethod
    def from_key_lists(cls, keys: List[str], prompt_template: str = "a medical article about {}", vectorizer=None):
        keys = sorted(set(keys))
        return cls(keys, prompt_template, vectorizer=vectorizer)

    def _encode_keys(self):
        texts = [self.prompt_template.format(key) for key in self.keys]
        vectors = self.vectorizer.transform(texts)
        self.text_features = np.array(vectors)
        norms = np.linalg.norm(self.text_features, axis=1, keepdims=True)
        self.text_features = self.text_features / np.where(norms == 0, 1.0, norms)
        print(f"precoded {len(self.keys)} key (: {self.text_features.shape[1]})", flush=True)

    def get_key_vector(self, key: str) -> np.ndarray:
        if key in self.keys:
            idx = self.keys.index(key)
            return self.text_features[idx]
        return None

    def get_all_keys(self) -> List[str]:
        return self.keys.copy()
