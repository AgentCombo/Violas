from typing import List
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster


# ================= Vector feature extraction method =================
class Vectorizer:
    """（ TF-IDF、Sentence Transformers Longformer）"""

    def __init__(self, method='tfidf', max_features=1024, n_components=None, model_name='all-MiniLM-L6-v2', model_paths_dict=None):
        """
        Args: method: ('tfidf', 'sentence_transformers', or 'longformer')
            max_features: TF-IDF n_components: （） model_name: （） model_paths_dict: （）
        """
        self.method = method
        self.max_features = max_features
        self.n_components = n_components
        self.model_name = model_name
        self.model_paths_dict = model_paths_dict or {}
        self.vectorizer = None
        self.reducer = None
        self.model = None  # for sentence_transformers

    def fit_transform(self, texts):
        """Run the benchmark helper."""
        if self.method == 'tfidf':
            # TF-IDF vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # Use 1-gram and 2-gram
                min_df=2,  # Minimum document frequency
                max_df=0.95,  # Maximum document frequency
                stop_words='english'  # Use English stop words
            )
            vectors = self.vectorizer.fit_transform(texts).toarray()

        elif self.method == 'sentence_transformers':
            # Sentence Transformers Vectorization
            vectors = self._fit_transform_sentence_transformers(texts)

        elif self.method == 'longformer':
            # Longformer vectorization
            vectors = self._fit_transform_longformer(texts)

        else:
            raise ValueError(f"Unsupported vectorizer method: {self.method}. Expected one of: 'tfidf', 'sentence_transformers', or 'longformer'")

        # Dimensionality reduction
        if self.n_components and self.n_components < vectors.shape[1]:
            print(f"SVD : {vectors.shape[1]} → {self.n_components}")
            self.reducer = TruncatedSVD(n_components=self.n_components, random_state=42)
            vectors = self.reducer.fit_transform(vectors)

        return vectors

    def transform(self, texts):
        """（）"""
        if self.method == 'tfidf':
            if self.vectorizer is None:
                raise ValueError("， fit_transform")
            vectors = self.vectorizer.transform(texts).toarray()

        elif self.method == 'sentence_transformers':
            if self.model is None:
                raise ValueError("， fit_transform")
            vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        elif self.method == 'longformer':
            # Longformer uses the same encoding
            vectors = self._transform_longformer(texts)

        else:
            raise ValueError(f": {self.method}")

        # If there is a dimensionality reducer, apply the dimensionality reduction
        if self.reducer is not None:
            vectors = self.reducer.transform(vectors)

        return vectors

    def _transform_longformer(self, texts):
        """Longformer"""
        import torch

        device = next(self.model.parameters()).device
        vectors = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(vec)

        return np.array(vectors)

    def _fit_transform_sentence_transformers(self, texts):
        """Sentence Transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            import os

            # Automatically find local model paths
            local_path = None
            if self.model_paths_dict:
                # Try multiple possible key name matches
                possible_keys = [
                    self.model_name,  # Original name, such as "all-MiniLM-L6-v2"
                    f"sentence-transformers-{self.model_name}",  # add prefix
                    f"sentence-transformers/{self.model_name}",  # HF format
                ]

                for key in possible_keys:
                    if key in self.model_paths_dict:
                        local_path = self.model_paths_dict[key]
                        # Check if the path exists
                        if os.path.exists(local_path):
                            print(f"✓ : {key} -> {local_path}")
                            break
                        else:
                            print(f"warning: missing cached vector for {key}: {local_path}")
                            local_path = None

            # Load model
            if local_path:
                # Use local path
                print(f": {local_path}")
                self.model = SentenceTransformer(local_path)
            else:
                # Download from online (use mirror acceleration)
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                print(f"，: {self.model_name}")
                print(f": {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
                self.model = SentenceTransformer(self.model_name)

            # encoded text
            print("...")
            vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

            return vectors

        except ImportError:
            print(": sentence-transformers")
            print("Install it with: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"Sentence Transformers : {e}")
            raise

    def _fit_transform_longformer(self, texts):
        """Longformer"""
        try:
            from transformers import LongformerModel, LongformerTokenizer
            import torch
            import os

            # Automatically find local model paths
            local_path = None
            if self.model_paths_dict:
                # Try multiple possible key name matches
                possible_keys = [
                    self.model_name,  # Original name, such as "longformer-base-4096"
                    f"longformer-{self.model_name}",  # add prefix
                    f"longformer/{self.model_name}",  # HF format
                ]

                for key in possible_keys:
                    if key in self.model_paths_dict:
                        local_path = self.model_paths_dict[key]
                        # Check if the path exists
                        if os.path.exists(local_path):
                            print(f"✓ : {key} -> {local_path}")
                            break
                        else:
                            print(f"warning: missing cached vector for {key}: {local_path}")
                            local_path = None

            # Load model and tokenizer
            if local_path:
                # Use local path
                print(f"Longformer : {local_path}")
                self.model = LongformerModel.from_pretrained(local_path)
                self.tokenizer = LongformerTokenizer.from_pretrained(local_path)
            else:
                # Download from online (use mirror acceleration)
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                print(f"，: {self.model_name}")
                print(f": {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
                self.model = LongformerModel.from_pretrained(self.model_name)
                self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)

            # Set up device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()

            # Suppress padding warnings for Longformer
            import logging
            logging.getLogger("transformers.modeling_longformer").setLevel(logging.ERROR)

            print(f"Device: {device}")
            print("...")

            vectors = []
            batch_size = 8

            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                    batch_texts = texts[i:i+batch_size]

                    # Word segmentation and encoding
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=4096,  # The maximum length of Longformer
                        return_tensors='pt'
                    )

                    # Move to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Get model output
                    outputs = self.model(**inputs)

                    # Use representation of [CLS] token as document vector
                    batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    vectors.extend(batch_vectors)

            vectors = np.array(vectors)
            print(f"Longformer ，: {vectors.shape}")

            return vectors

        except ImportError:
            print(": transformers")
            print("Install it with: pip install transformers torch")
            raise
        except Exception as e:
            print(f"Longformer : {e}")
            raise


# ================= Vector pooling method =================
def pool_vectors(vectors: List[np.ndarray], pool_factor: int = 2, method='hierarchical') -> List[np.ndarray]:
    """Run the benchmark helper."""
    if len(vectors) <= 1 or pool_factor <= 1:
        return vectors

    if method == 'hierarchical':
        # Hierarchical clustering and pooling
        vectors_array = np.array(vectors)
        similarities = np.dot(vectors_array, vectors_array.T)
        distances = 1 - similarities

        Z = linkage(distances, method='ward')
        max_clusters = max(1, len(vectors) // pool_factor)
        cluster_labels = fcluster(Z, t=max_clusters, criterion='maxclust')

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(vectors[i])

        return [np.mean(cluster, axis=0) for cluster in clusters.values()]

    elif method == 'kmeans':
        # K-means pooling
        max_clusters = max(1, len(vectors) // pool_factor)
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        kmeans.fit(vectors)

        clusters = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(vectors[i])

        return [np.mean(cluster, axis=0) for cluster in clusters.values()]

    elif method == 'random':
        # random pooling
        indices = list(range(len(vectors)))
        random.shuffle(indices)
        chunk_size = len(vectors) // pool_factor
        pooled_vectors = []

        for i in range(0, len(vectors), chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            chunk_vectors = [vectors[j] for j in chunk_indices]
            pooled_vectors.append(np.mean(chunk_vectors, axis=0))

        return pooled_vectors

    else:
        return vectors
