import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Introduce your text tool
from text_preprocess import collect_text_data_from_folders
from text_vectorizer import Vectorizer

def extract_all_text_features_from_folders(root_path, vectorizer_method='tfidf', model_name='all-MiniLM-L6-v2', test_size=0.1, random_state=42, max_features=1024, n_components=256):
    """Extract text and segment it. The default is tfidf (fast, no need to download); optional sentence_transformers / longformer.
    max_features: TF-IDF maximum number of features. n_components: Dimensionality reduction target dimension (such as 256), None means no dimensionality reduction."""
    print(f"Removing from {root_path} Read text and preprocess...")
    all_data = collect_text_data_from_folders(root_path)

    print(f"\nInitialize the feature extractor: {vectorizer_method} ({model_name}), max_features={max_features}, n_components={n_components}...")
    vectorizer = Vectorizer(method=vectorizer_method, model_name=model_name, max_features=max_features, n_components=n_components)

    all_texts = [item[1] for folder in all_data for item in all_data[folder]]
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
        file_paths = [item[0] for item in items]
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
    """KeyPredictor， key （ vectorizer）。"""
    def __init__(self, keys: List[str], prompt_template: str = "an article about {}", vectorizer=None):
        self.keys = keys
        self.prompt_template = prompt_template
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = Vectorizer(method='sentence_transformers', model_name='all-MiniLM-L6-v2')
            self.vectorizer.fit_transform(["warm up"])
        self._encode_keys()

    @classmethod
    def from_key_lists(cls, keys: List[str], prompt_template: str = "an article about {}", vectorizer=None):
        keys = sorted(set(keys))
        return cls(keys, prompt_template, vectorizer=vectorizer)

    def _encode_keys(self):
        # Change the folder name into a prompt word, such as "alt.atheism" -> "an article about alt.atheism"
        texts = [self.prompt_template.format(key) for key in self.keys]
        vectors = self.vectorizer.transform(texts)
        self.text_features = np.array(vectors)

        # normalization
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
