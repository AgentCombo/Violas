"""
Ohsumed - build_vectormap_from_train_data - build_milvus_collection_from_train_data ( key_str mixed ) - build_python_flat_index_from_train_data - search_python_flat_index / search_python_flat_mixed - search_milvus_mixed - build_qdrant_collection_from_train_data / search_qdrant_mixed - build_chroma_collection_from_train_data / search_chroma_mixed
"""

import os
import sys
import time
import numpy as np
import chromadb
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from violas.core.bench_recall_utils import lookup_key_vector
from violas.storage import VectorMap, VectorGroup
from ohsumed_feature_extract import KeyPredictor
from violas.core.ivf_flat import build_ivf_flat_index as _build_ivf_flat, search_ivf_flat_index as _search_ivf_flat


def build_vectormap_from_train_data(train_vectors_by_folder, train_descriptions_by_folder, predictor: KeyPredictor, alpha=0.5):
    """
    VectorMap锛?key锛?20news 锛?    """
    print("\n=== Start building VectorMap (including predict key and storage)===")
    train_data_by_key = {}

    print("1: train ...")
    for folder, vectors in train_vectors_by_folder.items():
        descriptions = train_descriptions_by_folder.get(folder, [])
        if folder not in train_data_by_key:
            train_data_by_key[folder] = {'vectors': [], 'descriptions': []}
        for vec, desc in zip(vectors, descriptions):
            train_data_by_key[folder]['vectors'].append(vec)
            train_data_by_key[folder]['descriptions'].append(desc)

    print(f"Train The data has been grouped by folder name, totaling {len(train_data_by_key)} key")

    print("\n2: VectorMap...")
    vectormap = VectorMap()
    for key, data in train_data_by_key.items():
        if len(data['vectors']) == 0:
            continue
        vectors = data['vectors']
        descriptions = data['descriptions']
        rep_vec = np.mean(np.array(vectors), axis=0)
        group = VectorGroup(
            group_name=key,
            representative=rep_vec,
            rep_description='simple_mean',
            vectors=vectors,
            descriptions=descriptions,
            vector_type="text"
        )
        vectormap.insert_with_auto_cluster(key, group, metadata={"data_type": "text"}, alpha=alpha)

    vectormap.set_key_vectors_from_predictor(predictor)
    vectormap.build_rep_index(nlist=100, nprobe=10)
    print("VectorMap built successfully")
    return vectormap, train_data_by_key


def build_milvus_collection_from_train_data(train_data_by_key, collection_name="ohsumed", use_lite=True, host="localhost", port="19530"):
    """Milvus Collection锛?key_str mixed"""
    print("\n=== Start building the Milvus Collection ===")
    try:
        try:
            connections.disconnect("default")
        except Exception:
            pass
        if use_lite:
            db_path = "./milvus_ohsumed.db"
            if os.path.exists(db_path):
                os.remove(db_path)
            connections.connect("default", uri=db_path)
        else:
            connections.connect("default", host=host, port=port)
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return None, None

    all_vectors = []
    all_ids = []
    all_keys = []
    id_to_result = {}
    milvus_id = 0

    for key, data in train_data_by_key.items():
        vectors = data['vectors']
        descriptions = data.get('descriptions', [])
        for idx, vec in enumerate(vectors):
            all_vectors.append(vec.tolist() if isinstance(vec, np.ndarray) else vec)
            all_ids.append(milvus_id)
            all_keys.append(str(key))
            img_path = descriptions[idx].get('img_path') if idx < len(descriptions) else None
            id_to_result[milvus_id] = (img_path, key)
            milvus_id += 1

    if len(all_vectors) == 0:
        return None, None

    dim = len(all_vectors[0])
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="key_str", dtype=DataType.VARCHAR, max_length=256)
    ]
    schema = CollectionSchema(fields, "Ohsumed text vector collection")
    collection = Collection(collection_name, schema)
    collection.insert([all_ids, all_vectors, all_keys])
    collection.flush()
    collection.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
    collection.load()
    print(f"Milvus Build completed, inserted {len(all_vectors)} vectors")
    return collection, id_to_result


def search_milvus_mixed(collection, query_vector, query_key_vector, key_vectors, id_to_result, beta, top_k,
                        candidate_factor=80, search_params=None):
    """Milvus mixed_score"""
    if collection is None or id_to_result is None:
        return None
    if search_params is None:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    n = collection.num_entities
    if n == 0:
        return []
    N = min(top_k * candidate_factor, n)
    N = max(N, top_k)
    query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
    try:
        results = collection.search(
            [query_vec], "vector", search_params,
            limit=N, output_fields=["id", "key_str"],
        )
    except Exception as e:
        if "key_str" in str(e) or "output_fields" in str(e).lower():
            return None
        raise
    if not results or len(results[0]) == 0:
        return []
    hits = results[0]
    q_key = None
    if query_key_vector is not None and beta != 0.0:
        q_key = np.asarray(query_key_vector, dtype=np.float64).flatten()
        qk_norm = np.linalg.norm(q_key)
        if np.isclose(qk_norm, 0.0):
            qk_norm = 1.0
        q_key = q_key / qk_norm
    scored = []
    for hit in hits:
        emb_sim = float(hit.distance)
        emb_d = 1.0 - emb_sim
        key = getattr(hit, "entity", None) and hit.entity.get("key_str")
        if key is None and hasattr(hit, "get"):
            key = hit.get("key_str")
        if key is None:
            key = id_to_result.get(hit.id, (None, None))[1]
        if key is None:
            scored.append((emb_d, hit.id))
            continue
        key = str(key).strip()
        if q_key is None or beta == 0.0:
            mixed = emb_d
        else:
            kv = lookup_key_vector(key, key_vectors)
            if kv is None:
                sem_d = 0.0
            else:
                kv = np.asarray(kv, dtype=np.float64).flatten()
                kv_norm = np.linalg.norm(kv)
                if np.isclose(kv_norm, 0.0):
                    kv_norm = 1.0
                kv = kv / kv_norm
                sem_d = 1.0 - float(np.dot(q_key, kv))
            mixed = beta * sem_d + (1.0 - beta) * emb_d
        scored.append((mixed, hit.id))
    scored.sort(key=lambda x: x[0])
    out = []
    for _, mid in scored[:top_k]:
        if mid in id_to_result:
            out.append(id_to_result[mid])
    return out


def build_qdrant_collection_from_train_data(train_data_by_key, collection_name="ohsumed", use_memory=True, host="localhost", port=6333):
    """Qdrant Collection"""
    print("\n=== Start building Qdrant Collection ===")
    if use_memory:
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(host=host, port=port)
    sample_vec = next(iter(train_data_by_key.values()))['vectors'][0]
    dim = len(sample_vec)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )
    points = []
    id_to_result = {}
    qdrant_id = 0
    for key, data in train_data_by_key.items():
        vectors = data['vectors']
        descriptions = data.get('descriptions', [])
        for idx, vec in enumerate(vectors):
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
            img_path = descriptions[idx].get('img_path') if idx < len(descriptions) else None
            id_to_result[qdrant_id] = (img_path, key)
            points.append(qmodels.PointStruct(id=qdrant_id, vector=vec_list, payload={"key": key}))
            qdrant_id += 1
    if not points:
        return None, None
    client.upsert(collection_name=collection_name, points=points)
    print(f"Qdrant Build completed, inserted {len(points)} vectors")
    return client, id_to_result


def search_qdrant_mixed(client, collection_name, query_vector, query_key_vector, key_vectors, id_to_result, beta, top_k, candidate_factor=80):
    """Qdrant mixed_score"""
    if client is None or id_to_result is None:
        return None
    N = max(top_k * candidate_factor, top_k)
    query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
    hits = client.query_points(collection_name=collection_name, query=query_vec, limit=N, with_payload=True).points
    if not hits:
        return []
    q_key = None
    if query_key_vector is not None and beta != 0.0:
        q_key = np.asarray(query_key_vector, dtype=np.float64).flatten()
        qk_norm = np.linalg.norm(q_key)
        if np.isclose(qk_norm, 0.0):
            qk_norm = 1.0
        q_key = q_key / qk_norm
    scored = []
    for hit in hits:
        emb_d = 1.0 - float(hit.score)
        key = hit.payload.get("key") if hit.payload else None
        if key is None:
            key = id_to_result.get(hit.id, (None, None))[1]
        if key is None:
            scored.append((emb_d, hit.id))
            continue
        key = str(key).strip()
        if q_key is None or beta == 0.0:
            mixed = emb_d
        else:
            kv = lookup_key_vector(key, key_vectors)
            if kv is None:
                sem_d = 0.0
            else:
                kv = np.asarray(kv, dtype=np.float64).flatten()
                kv_norm = np.linalg.norm(kv)
                if np.isclose(kv_norm, 0.0):
                    kv_norm = 1.0
                kv = kv / kv_norm
                sem_d = 1.0 - float(np.dot(q_key, kv))
            mixed = beta * sem_d + (1.0 - beta) * emb_d
        scored.append((mixed, hit.id))
    scored.sort(key=lambda x: x[0])
    return [id_to_result[qid] for _, qid in scored[:top_k] if qid in id_to_result]


def build_chroma_collection_from_train_data(train_data_by_key, collection_name="ohsumed"):
    """Chroma Collection ()"""
    print("\n=== Chroma Collection () ===")
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    ids = []
    embeddings = []
    metadatas = []
    id_to_result = {}
    idx_counter = 0
    for key, data in train_data_by_key.items():
        vectors = data['vectors']
        descriptions = data.get('descriptions', [])
        for i, vec in enumerate(vectors):
            str_id = f"id_{idx_counter}"
            img_path = descriptions[i].get('img_path') if i < len(descriptions) else None
            ids.append(str_id)
            embeddings.append(vec.tolist() if isinstance(vec, np.ndarray) else vec)
            metadatas.append({"key": str(key), "img_path": str(img_path)})
            id_to_result[str_id] = (img_path, key)
            idx_counter += 1
    batch_size = 2000
    for i in tqdm(range(0, len(ids), batch_size), desc="Chroma writing progress"):
        end_idx = min(i + batch_size, len(ids))
        collection.add(ids=ids[i:end_idx], embeddings=embeddings[i:end_idx], metadatas=metadatas[i:end_idx])
    print(f"Chroma Build completed, inserted {len(ids)} vectors")
    return collection, id_to_result


def search_chroma_mixed(collection, query_vector, query_key_vector, key_vectors, id_to_result, beta, top_k, candidate_factor=80):
    """Build a Chroma collection from training data."""
    if collection is None or id_to_result is None:
        return None
    N = max(top_k * candidate_factor, top_k)
    query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
    results = collection.query(query_embeddings=[query_vec], n_results=N, include=['metadatas', 'distances'])
    hits_ids = results['ids'][0]
    hits_distances = results['distances'][0]
    hits_metadatas = results['metadatas'][0]
    q_key = None
    if query_key_vector is not None and beta != 0.0:
        q_key = np.asarray(query_key_vector, dtype=np.float64).flatten()
        q_key /= (np.linalg.norm(q_key) + 1e-9)
    scored = []
    for i in range(len(hits_ids)):
        emb_d = float(hits_distances[i])
        key = hits_metadatas[i]['key']
        if q_key is None or beta == 0.0:
            mixed_score = emb_d
        else:
            kv = lookup_key_vector(key, key_vectors)
            if kv is None:
                sem_d = 0.5
            else:
                kv = np.asarray(kv, dtype=np.float64).flatten()
                kv /= (np.linalg.norm(kv) + 1e-9)
                sem_d = 1.0 - float(np.dot(q_key, kv))
            mixed_score = beta * sem_d + (1.0 - beta) * emb_d
        scored.append((mixed_score, hits_ids[i]))
    scored.sort(key=lambda x: x[0])
    return [id_to_result[sid] for _, sid in scored[:top_k] if sid in id_to_result]


def build_python_flat_index_from_train_data(train_data_by_key):
    """train_data_by_key Python Flat"""
    all_vectors = []
    img_paths = []
    keys = []
    for key, data in train_data_by_key.items():
        vectors = data.get("vectors", [])
        descriptions = data.get("descriptions", [])
        for idx, vec in enumerate(vectors):
            arr = np.asarray(vec, dtype=np.float32).flatten()
            all_vectors.append(arr)
            img_paths.append(descriptions[idx].get("img_path") if idx < len(descriptions) else None)
            keys.append(key)
    if len(all_vectors) == 0:
        return None
    matrix = np.vstack(all_vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    matrix = matrix / norms

    # Use Flat full table scan (without IVF), consistent with the paper
    ivf_centers, ivf_buckets = None, []
    ivf_nprobe = 16
    return {
        "vectors": matrix,
        "img_paths": img_paths,
        "keys": keys,
        "ivf_centers": ivf_centers,
        "ivf_buckets": ivf_buckets,
        "ivf_nprobe": ivf_nprobe,
    }


def search_python_flat_index(index_data, query_vector, top_k=10, mode="naive"):
    """Build a Python IVF-FLAT baseline index."""
    if index_data is None:
        return []
    matrix = index_data["vectors"]
    if matrix.shape[0] == 0:
        return []
    if index_data.get("ivf_centers") is not None and len(index_data.get("ivf_buckets", [])) > 0:
        nprobe = index_data.get("ivf_nprobe", 16)
        return _search_ivf_flat(
            matrix, index_data["img_paths"], index_data["keys"],
            index_data["ivf_centers"], index_data["ivf_buckets"],
            query_vector, top_k=top_k, nprobe=nprobe,
        )
    q = np.asarray(query_vector, dtype=np.float32).flatten()
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        q_norm = 1.0
    q = q / q_norm
    distances = 1.0 - (matrix @ q)
    k = min(top_k, len(distances))
    if k <= 0:
        return []
    if k == len(distances):
        top_idx = np.argsort(distances)
    else:
        part = np.argpartition(distances, k - 1)[:k]
        top_idx = part[np.argsort(distances[part])]
    return [(index_data["img_paths"][i], index_data["keys"][i], float(distances[i])) for i in top_idx]


def search_python_flat_mixed(index_data, query_vector, query_key_vector, key_vectors, beta=0.0, top_k=10):
    """锛?Ground Truth"""
    if index_data is None:
        return []
    matrix = index_data["vectors"]
    if matrix.shape[0] == 0:
        return []
    q = np.asarray(query_vector, dtype=np.float32).flatten()
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        q_norm = 1.0
    q = q / q_norm
    emb_dists = 1.0 - (matrix @ q)
    if beta == 0.0 or query_key_vector is None:
        mixed_scores = emb_dists
    else:
        q_key = np.asarray(query_key_vector, dtype=np.float32).flatten()
        qk_norm = np.linalg.norm(q_key)
        if np.isclose(qk_norm, 0.0):
            qk_norm = 1.0
        q_key = q_key / qk_norm
        keys_list = index_data["keys"]
        unique_keys = list(set(keys_list))
        key_sem_d = {}
        for k in unique_keys:
            kv = lookup_key_vector(k, key_vectors)
            if kv is not None:
                kv_arr = np.asarray(kv, dtype=np.float32).flatten()
                kv_norm = np.linalg.norm(kv_arr)
                if np.isclose(kv_norm, 0.0):
                    kv_norm = 1.0
                kv_arr = kv_arr / kv_norm
                key_sem_d[k] = 1.0 - float(np.dot(q_key, kv_arr))
            else:
                key_sem_d[k] = 0.0
        sem_dists = np.array([key_sem_d.get(k, 0.0) for k in keys_list], dtype=np.float32)
        mixed_scores = beta * sem_dists + (1.0 - beta) * emb_dists
    k = min(top_k, len(mixed_scores))
    if k <= 0:
        return []
    if k == len(mixed_scores):
        top_idx = np.argsort(mixed_scores)
    else:
        part = np.argpartition(mixed_scores, k - 1)[:k]
        top_idx = part[np.argsort(mixed_scores[part])]
    return [(index_data["img_paths"][i], index_data["keys"][i], float(mixed_scores[i])) for i in top_idx]
