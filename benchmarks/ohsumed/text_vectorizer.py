from typing import List
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster


class Vectorizer:
    """（ TF-IDF、Sentence Transformers Longformer）"""

    def __init__(self, method='tfidf', max_features=1024, n_components=None, model_name='all-MiniLM-L6-v2', model_paths_dict=None):
        self.method = method
        self.max_features = max_features
        self.n_components = n_components
        self.model_name = model_name
        self.model_paths_dict = model_paths_dict or {}
        self.vectorizer = None
        self.reducer = None
        self.model = None

    def fit_transform(self, texts):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            vectors = self.vectorizer.fit_transform(texts).toarray()
        elif self.method == 'sentence_transformers':
            vectors = self._fit_transform_sentence_transformers(texts)
        elif self.method == 'longformer':
            vectors = self._fit_transform_longformer(texts)
        else:
            raise ValueError(f": {self.method}")

        if self.n_components and self.n_components < vectors.shape[1]:
            print(f"SVD : {vectors.shape[1]} → {self.n_components}")
            self.reducer = TruncatedSVD(n_components=self.n_components, random_state=42)
            vectors = self.reducer.fit_transform(vectors)
        return vectors

    def transform(self, texts):
        if self.method == 'tfidf':
            if self.vectorizer is None:
                raise ValueError("， fit_transform")
            vectors = self.vectorizer.transform(texts).toarray()
        elif self.method == 'sentence_transformers':
            if self.model is None:
                raise ValueError("， fit_transform")
            vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        elif self.method == 'longformer':
            vectors = self._transform_longformer(texts)
        else:
            raise ValueError(f": {self.method}")

        if self.reducer is not None:
            vectors = self.reducer.transform(vectors)
        return vectors

    def _transform_longformer(self, texts):
        import torch
        device = next(self.model.parameters()).device
        vectors = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer([text], padding=True, truncation=True, max_length=4096, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(vec)
        return np.array(vectors)

    def _fit_transform_sentence_transformers(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
            local_path = None
            if self.model_paths_dict:
                for key in [self.model_name, f"sentence-transformers-{self.model_name}", f"sentence-transformers/{self.model_name}"]:
                    if key in self.model_paths_dict and os.path.exists(self.model_paths_dict[key]):
                        local_path = self.model_paths_dict[key]
                        break
            if local_path:
                self.model = SentenceTransformer(local_path)
            else:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.model = SentenceTransformer(self.model_name)
            return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        except ImportError:
            raise ImportError(": pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Sentence Transformers : {e}") from e

    def _fit_transform_longformer(self, texts):
        try:
            from transformers import LongformerModel, LongformerTokenizer
            import torch
            local_path = None
            if self.model_paths_dict:
                for key in [self.model_name, f"longformer-{self.model_name}", f"longformer/{self.model_name}"]:
                    if key in self.model_paths_dict and os.path.exists(self.model_paths_dict[key]):
                        local_path = self.model_paths_dict[key]
                        break
            if local_path:
                self.model = LongformerModel.from_pretrained(local_path)
                self.tokenizer = LongformerTokenizer.from_pretrained(local_path)
            else:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.model = LongformerModel.from_pretrained(self.model_name)
                self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            vectors = []
            batch_size = 8
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                    batch = texts[i:i+batch_size]
                    inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=4096, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    vectors.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return np.array(vectors)
        except ImportError:
            raise ImportError(": pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Longformer : {e}") from e


def pool_vectors(vectors: List[np.ndarray], pool_factor: int = 2, method='hierarchical') -> List[np.ndarray]:
    if len(vectors) <= 1 or pool_factor <= 1:
        return vectors
    if method == 'hierarchical':
        vectors_array = np.array(vectors)
        similarities = np.dot(vectors_array, vectors_array.T)
        distances = 1 - similarities
        Z = linkage(distances, method='ward')
        max_clusters = max(1, len(vectors) // pool_factor)
        cluster_labels = fcluster(Z, t=max_clusters, criterion='maxclust')
        clusters = {}
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(vectors[i])
        return [np.mean(c, axis=0) for c in clusters.values()]
    elif method == 'kmeans':
        max_clusters = max(1, len(vectors) // pool_factor)
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        kmeans.fit(vectors)
        clusters = {}
        for i, label in enumerate(kmeans.labels_):
            clusters.setdefault(label, []).append(vectors[i])
        return [np.mean(c, axis=0) for c in clusters.values()]
    elif method == 'random':
        indices = list(range(len(vectors)))
        random.shuffle(indices)
        chunk_size = max(1, len(vectors) // pool_factor)
        return [np.mean([vectors[j] for j in indices[i:i+chunk_size]], axis=0) for i in range(0, len(vectors), chunk_size)]
    return vectors
