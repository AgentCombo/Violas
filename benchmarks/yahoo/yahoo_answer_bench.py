"""
Yahoo! Answers - beta mixed GT锛坰earch_python_flat_mixed锛?- 锛欻DMG銆丷epresentative銆丳ythonFlat銆丮ilvus銆丵drant銆丆hroma
"""

import os
import sys
import re
import time
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from violas.storage import VectorMap
from yahoo_feature_extract import extract_all_text_features_from_folders, KeyPredictor
from database_build import (
    build_vectormap_from_train_data,
    build_milvus_collection_from_train_data,
    build_python_flat_index_from_train_data,
    search_python_flat_index,
    search_python_flat_mixed,
    search_milvus_mixed,
    build_qdrant_collection_from_train_data,
    search_qdrant_mixed,
    build_chroma_collection_from_train_data,
    search_chroma_mixed,
)
from violas.core.bench_recall_utils import (
    is_beta_pure_key,
    norm_key_for_recall,
    padded_norm_keys_from_search_results,
    padded_norm_keys_from_tuples,
    recall_key_hit_rate_at_k,
)
from violas.core.legacy_pickle import enable_legacy_storage_pickle

_predictor = None
_python_flat_index = None
_python_flat_mode = "naive"


def build_vector_database_for_test(
    root,
    test_size=0.1,
    random_state=42,
    build_milvus=False,
    use_milvus_lite=True,
    milvus_host="localhost",
    milvus_port="19530",
    build_qdrant=False,
    build_chroma=False,
    alpha=0.5,
    use_cache=True,
    cache_prefix=None,
    sample_ratio=None,
    max_files_per_folder=None,
    max_features=256,
    vectorizer_method='tfidf',
):
    """銆俶ax_features: TF-IDF 锛?256銆?""
    global _predictor, _python_flat_index

    print("=== VectorMap ===")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache_dir = os.path.join(project_root, "cache")
    legacy_cache_dir = os.path.abspath(os.path.join(project_root, "..", "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    if cache_prefix is None:
        dataset_name = os.path.basename(os.path.normpath(root))
        cache_prefix = re.sub(r"[^0-9a-zA-Z._-]+", "_", dataset_name)
    cache_name = f"yahoo_{cache_prefix}_ts{test_size}_rs{random_state}_a{alpha:.3f}_mf{max_features}_vm{vectorizer_method}_folder.pkl"
    cache_file = os.path.join(cache_dir, cache_name)
    cache_candidates = [
        cache_file,
        os.path.join(legacy_cache_dir, cache_name),
        os.path.join(legacy_cache_dir, f"yahoo_{cache_prefix}_ts{test_size}_rs{random_state}_a{alpha:.3f}_mf{max_features}_folder.pkl"),
    ]
    for candidate in cache_candidates:
        if os.path.exists(candidate):
            cache_file = candidate
            break

    vectormap = None
    test_data = None
    train_data_by_key = None
    folders = None

    if use_cache and os.path.exists(cache_file):
        print(f"\n[Cache] Hit cache: {cache_file}")
        with open(cache_file, "rb") as f:
            enable_legacy_storage_pickle()
            cache_obj = pickle.load(f)
        vectormap = cache_obj["vectormap"]
        test_data = cache_obj["test_data"]
        train_data_by_key = cache_obj["train_data_by_key"]
        folders = cache_obj["folders"]
        _predictor = KeyPredictor.from_key_lists(folders, vectorizer=cache_obj.get("vectorizer"))
        print(f"[Cache] VectorMap/TestData loaded, number of keys: {len(_predictor.get_all_keys())}")
    else:
        print(f"\nStep 1: 锛坽vectorizer_method}锛?..")
        train_vectors_by_folder, train_descriptions_by_folder, test_data, _vectorizer = extract_all_text_features_from_folders(
            root,
            vectorizer_method=vectorizer_method,
            test_size=test_size,
            random_state=random_state,
            sample_ratio=sample_ratio,
            max_files_per_folder=max_files_per_folder,
            max_features=max_features,
        )
        folders = sorted(list(train_vectors_by_folder.keys()))
        print(f"Collected in total {len(folders)} folder data")

        print("\nStep 2: KeyPredictor锛?vectorizer锛?..")
        _predictor = KeyPredictor.from_key_lists(folders, vectorizer=_vectorizer)

        vectormap, train_data_by_key = build_vectormap_from_train_data(
            train_vectors_by_folder,
            train_descriptions_by_folder,
            _predictor,
            alpha=alpha
        )

        if use_cache:
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "vectormap": vectormap,
                    "test_data": test_data,
                    "train_data_by_key": train_data_by_key,
                    "folders": folders,
                    "vectorizer": _vectorizer,
                }, f)
            print(f"[Cache] Written to cache: {cache_file}")

    milvus_collection, milvus_id_map = None, None
    if build_milvus:
        milvus_collection, milvus_id_map = build_milvus_collection_from_train_data(
            train_data_by_key,
            collection_name="yahoo_answer",
            use_lite=use_milvus_lite,
            host=milvus_host,
            port=milvus_port,
        )

    qdrant_client, qdrant_id_map = None, None
    if build_qdrant:
        qdrant_client, qdrant_id_map = build_qdrant_collection_from_train_data(
            train_data_by_key,
            collection_name="yahoo_answer",
        )

    chroma_collection, chroma_id_map = None, None
    if build_chroma:
        chroma_collection, chroma_id_map = build_chroma_collection_from_train_data(
            train_data_by_key,
            collection_name="yahoo_answer",
        )

    _python_flat_index = build_python_flat_index_from_train_data(train_data_by_key)
    if _python_flat_index is not None:
        print(f"PythonFlat Index built, number of vectors: {len(_python_flat_index['keys'])}", flush=True)

    return vectormap, test_data, milvus_collection, milvus_id_map, qdrant_client, qdrant_id_map, chroma_collection, chroma_id_map


def test_recall(
    vectormap,
    test_data,
    top_k=10,
    milvus_collection=None,
    milvus_id_map=None,
    qdrant_client=None,
    qdrant_id_map=None,
    chroma_collection=None,
    chroma_id_map=None,
    save_results=True,
    output_dir="outputs/yahoo",
    expansion_factor=10,
    beta=0.5,
    file_prefix=None,
    alpha=None,
    use_hdmg=False,
    hdmg_cluster_pool_size=64,
    hdmg_extra_hops=0,
    max_queries=None,
):
    """Recall 锛?20news 锛夈€俶ax_queries: 锛孨one 銆?""
    global _predictor, _python_flat_index

    def get_result_id(result):
        if result.vector_idx is not None and result.vector_idx < len(result.group.descriptions):
            return result.group.descriptions[result.vector_idx].get('img_path')
        return None

    all_test_samples = []
    for folder, data in test_data.items():
        vectors = data['vectors']
        descriptions = data['descriptions']
        for vec, desc in zip(vectors, descriptions):
            all_test_samples.append({'folder': folder, 'vector': vec, 'description': desc})

    if len(all_test_samples) == 0:
        print("warning: no successful queries")
        return {"average_recall_mixed": 0.0}

    if max_queries is not None and max_queries > 0:
        all_test_samples = all_test_samples[:max_queries]
        print(f"Only before testing {len(all_test_samples)} (max_queries={max_queries})")

    recalls_mixed = []
    recalls_representative = []
    recalls_python_flat = []
    recalls_milvus = []
    recalls_milvus_mixed = []
    recalls_qdrant = []
    recalls_qdrant_mixed = []
    recalls_chroma = []
    recalls_chroma_mixed = []

    total_gt_time = total_mixed_time = total_representative_time = 0.0
    total_python_flat_time = total_milvus_time = total_milvus_mixed_time = 0.0
    total_qdrant_time = total_qdrant_mixed_time = total_chroma_time = total_chroma_mixed_time = 0.0
    successful_queries = 0
    detailed_results = []

    for idx, sample in enumerate(tqdm(all_test_samples, desc="Under test"), 1):
        query_vector = sample['vector']
        query_folder = sample['folder']
        predicted_key = query_folder
        query_key_vector = _predictor.get_key_vector(query_folder)

        try:
            gt_start = time.time()
            gt_mixed_results = search_python_flat_mixed(
                _python_flat_index, query_vector, query_key_vector,
                vectormap.key_vectors, beta=beta, top_k=top_k
            )
            gt_end = time.time()
            total_gt_time += (gt_end - gt_start)
            gt_img_paths = [r[0] for r in gt_mixed_results]

            mixed_start = time.time()
            if use_hdmg:
                mixed_results = vectormap.search_hdmg(
                    query_vector,
                    query_key_vector=query_key_vector,
                    alpha=beta,
                    top_k=top_k,
                    max_steps=100,
                    distance_method="cosine",
                    entry_alpha_threshold=0.3,
                    cluster_pool_size=hdmg_cluster_pool_size,
                    extra_hops=hdmg_extra_hops,
                )
            else:
                mixed_results = vectormap.search_with_mixed_key_rep_vec(
                    query_vector,
                    query_key_vector=query_key_vector,
                    beta=beta,
                    top_k=top_k,
                    gruop_expansion_factor=expansion_factor,
                    distance_method="cosine",
                )
            mixed_end = time.time()
            total_mixed_time += (mixed_end - mixed_start)

            representative_start = time.time()
            representative_results = vectormap.search_with_representative_rerank(
                query_vector,
                query_key_vector=query_key_vector,
                beta=beta,
                top_k=top_k,
                num_groups=top_k,
                distance_method="cosine",
            )
            representative_end = time.time()
            total_representative_time += (representative_end - representative_start)

            python_flat_results = None
            python_start = python_end = None
            if _python_flat_index is not None:
                python_start = time.time()
                python_flat_results = search_python_flat_index(
                    _python_flat_index, query_vector, top_k=top_k, mode=_python_flat_mode
                )
                python_end = time.time()
                total_python_flat_time += (python_end - python_start)

            milvus_results = None
            milvus_mixed_results = None
            milvus_start = milvus_end = milvus_mixed_start = milvus_mixed_end = None
            if milvus_collection is not None and milvus_id_map is not None:
                milvus_start = time.time()
                query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
                sr = milvus_collection.search(
                    [query_vec], "vector",
                    {"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=top_k, output_fields=["id"]
                )
                milvus_end = time.time()
                total_milvus_time += (milvus_end - milvus_start)
                if len(sr) > 0:
                    milvus_results = [milvus_id_map[hit.id] for hit in sr[0] if hit.id in milvus_id_map]

                milvus_mixed_start = time.time()
                milvus_mixed_results = search_milvus_mixed(
                    milvus_collection, query_vector, query_key_vector,
                    vectormap.key_vectors, milvus_id_map, beta=beta, top_k=top_k, candidate_factor=10,
                )
                milvus_mixed_end = time.time()
                if milvus_mixed_results is not None:
                    total_milvus_mixed_time += (milvus_mixed_end - milvus_mixed_start)

            qdrant_results = None
            qdrant_mixed_results = None
            q_start = q_end = qdrant_mixed_start = qdrant_mixed_end = None
            if qdrant_client is not None and qdrant_id_map is not None:
                q_start = time.time()
                query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
                q_res = qdrant_client.query_points(collection_name="yahoo_answer", query=query_vec, limit=top_k)
                q_hits = q_res.points
                q_end = time.time()
                total_qdrant_time += (q_end - q_start)
                qdrant_results = [qdrant_id_map[hit.id] for hit in q_hits if hit.id in qdrant_id_map]

                qdrant_mixed_start = time.time()
                qdrant_mixed_results = search_qdrant_mixed(
                    qdrant_client, "yahoo_answer",
                    query_vector, query_key_vector, vectormap.key_vectors,
                    qdrant_id_map, beta=beta, top_k=top_k, candidate_factor=10,
                )
                qdrant_mixed_end = time.time()
                if qdrant_mixed_results is not None:
                    total_qdrant_mixed_time += (qdrant_mixed_end - qdrant_mixed_start)

            chroma_results = None
            chroma_mixed_results = None
            c_start = c_end = cm_start = cm_end = None
            if chroma_collection is not None and chroma_id_map is not None:
                c_start = time.time()
                chroma_results = search_chroma_mixed(
                    chroma_collection, query_vector, query_key_vector, vectormap.key_vectors,
                    chroma_id_map, beta=0.0, top_k=top_k, candidate_factor=1
                )
                c_end = time.time()
                total_chroma_time += (c_end - c_start)

                cm_start = time.time()
                chroma_mixed_results = search_chroma_mixed(
                    chroma_collection, query_vector, query_key_vector, vectormap.key_vectors,
                    chroma_id_map, beta=beta, top_k=top_k, candidate_factor=10
                )
                cm_end = time.time()
                total_chroma_mixed_time += (cm_end - cm_start)

            nk = norm_key_for_recall
            use_key_recall = is_beta_pure_key(beta)
            ord_mixed = ord_rep = ord_pf = None
            ord_milvus = ord_milvus_m = ord_q = ord_qm = ord_c = ord_cm = None
            if use_key_recall:
                gt_ids = set(nk(r[1]) for r in gt_mixed_results if r is not None and len(r) > 1 and r[1] is not None)
                ord_mixed = padded_norm_keys_from_search_results(mixed_results, nk, top_k)
                ord_rep = padded_norm_keys_from_search_results(representative_results, nk, top_k)
                recall_mixed = recall_key_hit_rate_at_k(gt_ids, ord_mixed, top_k)
                recalls_mixed.append(recall_mixed)
                recall_representative = recall_key_hit_rate_at_k(gt_ids, ord_rep, top_k)
                recalls_representative.append(recall_representative)
            else:
                gt_ids = set(p for p in gt_img_paths if p is not None)
                mixed_ids = set(id for id in (get_result_id(r) for r in mixed_results) if id is not None)
                representative_ids = set(id for id in (get_result_id(r) for r in representative_results) if id is not None)
                recall_mixed = len(gt_ids & mixed_ids) / len(gt_ids) if gt_ids else 0.0
                recalls_mixed.append(recall_mixed)
                recall_representative = len(gt_ids & representative_ids) / len(gt_ids) if gt_ids else 0.0
                recalls_representative.append(recall_representative)

            if python_flat_results is not None:
                if use_key_recall:
                    ord_pf = padded_norm_keys_from_tuples(python_flat_results, nk, top_k)
                    recalls_python_flat.append(recall_key_hit_rate_at_k(gt_ids, ord_pf, top_k))
                else:
                    python_img_paths = set(item[0] for item in python_flat_results if item[0] is not None)
                    recalls_python_flat.append(len(gt_ids & python_img_paths) / len(gt_ids) if gt_ids else 0.0)

            if milvus_results is not None:
                if use_key_recall:
                    ord_milvus = padded_norm_keys_from_tuples(milvus_results, nk, top_k)
                    recalls_milvus.append(recall_key_hit_rate_at_k(gt_ids, ord_milvus, top_k))
                else:
                    milvus_img_paths = set(item[0] for item in milvus_results if item[0] is not None)
                    recalls_milvus.append(len(gt_ids & milvus_img_paths) / len(gt_ids) if gt_ids else 0.0)
            if milvus_mixed_results is not None:
                if use_key_recall:
                    ord_milvus_m = padded_norm_keys_from_tuples(milvus_mixed_results, nk, top_k)
                    recalls_milvus_mixed.append(recall_key_hit_rate_at_k(gt_ids, ord_milvus_m, top_k))
                else:
                    milvus_mixed_img_paths = set(item[0] for item in milvus_mixed_results if item[0] is not None)
                    recalls_milvus_mixed.append(len(gt_ids & milvus_mixed_img_paths) / len(gt_ids) if gt_ids else 0.0)

            if qdrant_results is not None:
                if use_key_recall:
                    ord_q = padded_norm_keys_from_tuples(qdrant_results, nk, top_k)
                    recalls_qdrant.append(recall_key_hit_rate_at_k(gt_ids, ord_q, top_k))
                else:
                    qdrant_img_paths = set(item[0] for item in qdrant_results if item[0] is not None)
                    recalls_qdrant.append(len(gt_ids & qdrant_img_paths) / len(gt_ids) if gt_ids else 0.0)
            if qdrant_mixed_results is not None:
                if use_key_recall:
                    ord_qm = padded_norm_keys_from_tuples(qdrant_mixed_results, nk, top_k)
                    recalls_qdrant_mixed.append(recall_key_hit_rate_at_k(gt_ids, ord_qm, top_k))
                else:
                    qdrant_mixed_img_paths = set(item[0] for item in qdrant_mixed_results if item[0] is not None)
                    recalls_qdrant_mixed.append(len(gt_ids & qdrant_mixed_img_paths) / len(gt_ids) if gt_ids else 0.0)

            if chroma_results is not None:
                if use_key_recall:
                    ord_c = padded_norm_keys_from_tuples(chroma_results, nk, top_k)
                    recalls_chroma.append(recall_key_hit_rate_at_k(gt_ids, ord_c, top_k))
                else:
                    chroma_img_paths = set(item[0] for item in chroma_results if item[0] is not None)
                    recalls_chroma.append(len(gt_ids & chroma_img_paths) / len(gt_ids) if gt_ids else 0.0)
            if chroma_mixed_results is not None:
                if use_key_recall:
                    ord_cm = padded_norm_keys_from_tuples(chroma_mixed_results, nk, top_k)
                    recalls_chroma_mixed.append(recall_key_hit_rate_at_k(gt_ids, ord_cm, top_k))
                else:
                    chroma_mixed_img_paths = set(item[0] for item in chroma_mixed_results if item[0] is not None)
                    recalls_chroma_mixed.append(len(gt_ids & chroma_mixed_img_paths) / len(gt_ids) if gt_ids else 0.0)

            def get_results_with_img_paths(results):
                return [{'id': (r.key, r.group.group_name, r.vector_idx or -1), 'img_path': get_result_id(r), 'distance': float(r.distance)} for r in results]

            detailed_results.append({
                'query_id': idx, 'query_folder': query_folder, 'recall_mixed': recall_mixed,
                'recall_representative': recall_representative,
                'ground_truth': [{'img_path': r[0], 'key': r[1], 'distance': float(r[2])} for r in gt_mixed_results],
                'mixed_results': get_results_with_img_paths(mixed_results),
                'representative_results': get_results_with_img_paths(representative_results),
                'milvus_ids': [(item[0], item[1]) for item in milvus_results] if milvus_results is not None else None,
                'milvus_mixed_ids': [(item[0], item[1]) for item in milvus_mixed_results] if milvus_mixed_results is not None else None,
                'qdrant_ids': [(item[0], item[1]) for item in qdrant_results] if qdrant_results is not None else None,
                'qdrant_mixed_ids': [(item[0], item[1]) for item in qdrant_mixed_results] if qdrant_mixed_results is not None else None,
                'chroma_ids': [(item[0], item[1]) for item in chroma_results] if chroma_results is not None else None,
                'chroma_mixed_ids': [(item[0], item[1]) for item in chroma_mixed_results] if chroma_mixed_results is not None else None,
                'gt_time': gt_end - gt_start, 'mixed_time': mixed_end - mixed_start,
                'representative_time': representative_end - representative_start,
            })
            successful_queries += 1

        except Exception as e:
            print(f"\n(folder={query_folder}): {e}")

    if len(recalls_mixed) == 0:
        return {"average_recall_mixed": 0.0}

    n = successful_queries
    stats = {
        'average_recall_mixed': float(np.mean(recalls_mixed)),
        'average_recall_representative': float(np.mean(recalls_representative)),
        'average_recall_python_flat': float(np.mean(recalls_python_flat)) if recalls_python_flat else None,
        'average_recall_milvus': float(np.mean(recalls_milvus)) if recalls_milvus else None,
        'average_recall_milvus_mixed': float(np.mean(recalls_milvus_mixed)) if recalls_milvus_mixed else None,
        'average_recall_qdrant': float(np.mean(recalls_qdrant)) if recalls_qdrant else None,
        'average_recall_qdrant_mixed': float(np.mean(recalls_qdrant_mixed)) if recalls_qdrant_mixed else None,
        'average_recall_chroma': float(np.mean(recalls_chroma)) if recalls_chroma else None,
        'average_recall_chroma_mixed': float(np.mean(recalls_chroma_mixed)) if recalls_chroma_mixed else None,
        'min_recall_mixed': float(np.min(recalls_mixed)),
        'max_recall_mixed': float(np.max(recalls_mixed)),
        'avg_gt_time': total_gt_time / n,
        'avg_mixed_time': total_mixed_time / n,
        'avg_representative_time': total_representative_time / n,
        'avg_python_flat_time': total_python_flat_time / n if _python_flat_index else None,
        'avg_milvus_time': total_milvus_time / n if milvus_collection else None,
        'avg_milvus_mixed_time': total_milvus_mixed_time / n if milvus_collection else None,
        'avg_qdrant_time': total_qdrant_time / n if qdrant_client else None,
        'avg_qdrant_mixed_time': total_qdrant_mixed_time / n if qdrant_client else None,
        'avg_chroma_time': total_chroma_time / n if chroma_collection else None,
        'avg_chroma_mixed_time': total_chroma_mixed_time / n if chroma_collection else None,
    }

    print(f"\n=== Recall Test results ===")
    print(f"Mixed Recall@{top_k}: {stats['average_recall_mixed']:.4f}")
    print(f"Representative Recall@{top_k}: {stats['average_recall_representative']:.4f}")
    if stats['average_recall_python_flat'] is not None:
        print(f"PythonFlat Recall@{top_k}: {stats['average_recall_python_flat']:.4f}")
    if stats['average_recall_milvus'] is not None:
        print(f"Milvus Recall@{top_k}: {stats['average_recall_milvus']:.4f}")
    if stats['average_recall_milvus_mixed'] is not None:
        print(f"Milvus Mixed Recall@{top_k}: {stats['average_recall_milvus_mixed']:.4f}")
    if stats['average_recall_qdrant'] is not None:
        print(f"Qdrant Recall@{top_k}: {stats['average_recall_qdrant']:.4f}")
    if stats.get('average_recall_qdrant_mixed') is not None:
        print(f"Qdrant Mixed Recall@{top_k}: {stats['average_recall_qdrant_mixed']:.4f}")
    if stats['average_recall_chroma'] is not None:
        print(f"Chroma Recall@{top_k}: {stats['average_recall_chroma']:.4f}")
    if stats.get('average_recall_chroma_mixed') is not None:
        print(f"Chroma Mixed Recall@{top_k}: {stats['average_recall_chroma_mixed']:.4f}")

    print(f"\n--- (: /) ---")
    print(f"Ground Truth (brute-force mixed) Total time: {total_gt_time:.2f} , : {stats['avg_gt_time']*1000:.2f} ms")
    print(f"Mixed Total time: {total_mixed_time:.2f} , : {stats['avg_mixed_time']*1000:.2f} ms")
    print(f"Representative Total time: {total_representative_time:.2f} , : {stats['avg_representative_time']*1000:.2f} ms")
    if stats.get('avg_python_flat_time') is not None:
        print(f"PythonFlat Total time: {total_python_flat_time:.2f} , : {stats['avg_python_flat_time']*1000:.2f} ms")
    if stats.get('avg_milvus_time') is not None:
        print(f"Milvus Total time: {total_milvus_time:.2f} , : {stats['avg_milvus_time']*1000:.2f} ms")
    if stats.get('avg_milvus_mixed_time') is not None:
        print(f"Milvus Mixed Total time: {total_milvus_mixed_time:.2f} , : {stats['avg_milvus_mixed_time']*1000:.2f} ms")
    if stats.get('avg_qdrant_time') is not None:
        print(f"Qdrant Total time: {total_qdrant_time:.2f} , : {stats['avg_qdrant_time']*1000:.2f} ms")
    if stats.get('avg_qdrant_mixed_time') is not None:
        print(f"Qdrant Mixed Total time: {total_qdrant_mixed_time:.2f} , : {stats['avg_qdrant_mixed_time']*1000:.2f} ms")
    if stats.get('avg_chroma_time') is not None:
        print(f"Chroma Total time: {total_chroma_time:.2f} , : {stats['avg_chroma_time']*1000:.2f} ms")
    if stats.get('avg_chroma_mixed_time') is not None:
        print(f"Chroma Mixed Total time: {total_chroma_mixed_time:.2f} , : {stats['avg_chroma_mixed_time']*1000:.2f} ms")
    if total_gt_time > 0 and total_mixed_time > 0:
        print(f"\nMixed Speedup ratio: {total_gt_time/total_mixed_time:.2f}x (Ground Truth)")

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        prefix = file_prefix or datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(output_dir, f"recall_detailed_{prefix}_exp{expansion_factor}_beta{beta:.1f}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({
                'test_config': {'top_k': top_k, 'alpha': alpha, 'beta': beta, 'expansion_factor': expansion_factor},
                'statistics': {k: v for k, v in stats.items() if v is not None},
                'detailed_results': detailed_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Missing required artifact: {fn}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yahoo Answers recall benchmark")
    parser.add_argument("--sample_ratio", "-s", type=float, default=0.01, help="(0~1)锛?.01")
    parser.add_argument("--max_queries", "-n", type=int, default=200, help="锛?200锛?")
    parser.add_argument("--max_features", "-m", type=int, default=256, help="TF-IDF 锛?256锛夛紝 256")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved items")
    parser.add_argument("--alpha", type=float, default=0.5, help="Within-group clustering strength 0~1")
    parser.add_argument("--expansion_factor", type=int, default=3, help="candidate pool multiple")
    parser.add_argument("--hdmg_cluster_pool_size", type=int, default=9, help="HDMG enabled")
    parser.add_argument("--hdmg_embedding_k", type=int, default=12)
    parser.add_argument("--hdmg_semantic_intra_k", type=int, default=20)
    parser.add_argument("--hdmg_extra_hops", type=int, default=0)
    parser.add_argument("--root", default=None, help="锛?dataset/Yahoo!-answer_processed")
    parser.add_argument("--vectorizer_method", "-vm", default="tfidf", choices=["tfidf", "sentence_transformers"],
                        help="锛歵fidf sentence_transformers")
    parser.add_argument("--summary_json", default=None, help="If specified, write the parameter test summary and time comparison to the JSON file")
    parser.add_argument("--only_beta", type=float, default=None, help="If specified, only this beta will be run (overriding the default beta scan)")
    args = parser.parse_args()

    root = args.root or os.path.abspath(os.path.join(REPO_ROOT, "..", "dataset", "Yahoo!-answer_processed"))

    enable_external_dbs = os.environ.get("VIOLAS_ENABLE_EXTERNAL_DBS", "0") == "1"
    BUILD_MILVUS = enable_external_dbs
    USE_MILVUS_LITE = True
    BUILD_QDRANT = enable_external_dbs
    BUILD_CHROMA = enable_external_dbs

    TOP_K = args.top_k
    ALPHA = args.alpha
    BETA = 0.3  # For display only, multiple sets of betas will actually be scanned
    EXPANSION_FACTOR = args.expansion_factor
    HDMG_CLUSTER_POOL_SIZE = args.hdmg_cluster_pool_size
    HDMG_EMBEDDING_K = args.hdmg_embedding_k
    HDMG_SEMANTIC_INTRA_K = args.hdmg_semantic_intra_k
    HDMG_EXTRA_HOPS = args.hdmg_extra_hops

    SAMPLE_RATIO = args.sample_ratio
    MAX_QUERIES = args.max_queries if args.max_queries > 0 else None
    MAX_FEATURES = args.max_features
    if SAMPLE_RATIO <= 0 or SAMPLE_RATIO > 1:
        print(": sample_ratio (0, 1]")
        sys.exit(1)

    main_start = time.time()
    build_start = time.time()
    vectormap, test_data, milvus_collection, milvus_id_map, qdrant_client, qdrant_id_map, chroma_collection, chroma_id_map = build_vector_database_for_test(
        root,
        test_size=0.1,
        build_milvus=BUILD_MILVUS,
        use_milvus_lite=USE_MILVUS_LITE,
        build_qdrant=BUILD_QDRANT,
        build_chroma=BUILD_CHROMA,
        alpha=ALPHA,
        sample_ratio=SAMPLE_RATIO,
        max_features=MAX_FEATURES,
        vectorizer_method=args.vectorizer_method,
    )
    print(f"The build is completed and takes: {time.time() - build_start:.2f} Second")

    vectormap.build_hdmg(
        embedding_k=HDMG_EMBEDDING_K,
        semantic_intra_k=HDMG_SEMANTIC_INTRA_K,
        semantic_bridge_keys=2,
        semantic_bridge_per_key=1,
        use_mutual_embedding=False,
    )
    print("HDMG enabled")

    print(f"\n=== Test set statistics ===")
    vectormap.analyze_vectormap_storage()
    total_test = sum(len(d['vectors']) for d in test_data.values())
    print(f"Number of test set folders: {len(test_data)}")
    print(f"Total number of vectors in test set: {total_test}")

    param_combinations = (
        [args.only_beta]
        if args.only_beta is not None
        else [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
    )

    test_run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = []

    for beta in param_combinations:
        print(f"\n--- beta: {beta} ---")
        stats = test_recall(
            vectormap, test_data, top_k=TOP_K,
            milvus_collection=milvus_collection, milvus_id_map=milvus_id_map,
            qdrant_client=qdrant_client, qdrant_id_map=qdrant_id_map,
            chroma_collection=chroma_collection, chroma_id_map=chroma_id_map,
            expansion_factor=EXPANSION_FACTOR, beta=beta,
            file_prefix=test_run_prefix, alpha=ALPHA,
            use_hdmg=True, hdmg_cluster_pool_size=HDMG_CLUSTER_POOL_SIZE, hdmg_extra_hops=HDMG_EXTRA_HOPS,
            max_queries=MAX_QUERIES,
        )
        results_summary.append({'beta': beta, **stats})

    print(f"\n{'='*120}")
    print("=== (Recall) ===")
    print(f"{'beta':<6} | {'mixed':<8} {'representative':<14} {'pyflat':<8} {'milvus':<8} {'milvus_mix':<10} {'qdrant':<8} {'qdrant_mix':<10} {'chroma':<8} {'chroma_mix':<10}")
    print("-" * 120)
    for r in results_summary:
        pyflat = f"{r['average_recall_python_flat']:.4f}" if r.get('average_recall_python_flat') else "N/A"
        milvus = f"{r['average_recall_milvus']:.4f}" if r.get('average_recall_milvus') else "N/A"
        milvus_mix = f"{r.get('average_recall_milvus_mixed'):.4f}" if r.get('average_recall_milvus_mixed') else "N/A"
        qdrant = f"{r.get('average_recall_qdrant'):.4f}" if r.get('average_recall_qdrant') else "N/A"
        qdrant_mix = f"{r.get('average_recall_qdrant_mixed'):.4f}" if r.get('average_recall_qdrant_mixed') else "N/A"
        chroma = f"{r.get('average_recall_chroma'):.4f}" if r.get('average_recall_chroma') else "N/A"
        chroma_mix = f"{r.get('average_recall_chroma_mixed'):.4f}" if r.get('average_recall_chroma_mixed') else "N/A"
        print(f"{r['beta']:<6.2f} | {r['average_recall_mixed']:<8.4f} {r['average_recall_representative']:<14.4f} {pyflat:<8} {milvus:<8} {milvus_mix:<10} {qdrant:<8} {qdrant_mix:<10} {chroma:<8} {chroma_mix:<10}")
    print(f"\n=== (/) ===")
    print(f"{'beta':<6} | {'mixed':<10} {'representative':<14} {'pyflat':<10} {'milvus':<10} {'milvus_mix':<12} {'qdrant':<10} {'qdrant_mix':<12} {'chroma':<10} {'chroma_mix':<12}")
    print("-" * 120)
    for r in results_summary:
        def ms(v):
            return f"{v*1000:.2f}" if v is not None else "N/A"
        print(f"{r['beta']:<6.2f} | {ms(r.get('avg_mixed_time')):<10} {ms(r.get('avg_representative_time')):<14} {ms(r.get('avg_python_flat_time')):<10} {ms(r.get('avg_milvus_time')):<10} {ms(r.get('avg_milvus_mixed_time')):<12} {ms(r.get('avg_qdrant_time')):<10} {ms(r.get('avg_qdrant_mixed_time')):<12} {ms(r.get('avg_chroma_time')):<10} {ms(r.get('avg_chroma_mixed_time')):<12}")
    print(f"{'='*120}")
    if getattr(args, "summary_json", None):
        betas = [r["beta"] for r in results_summary]
        recall = {
            "mixed": [r.get("average_recall_mixed") for r in results_summary],
            "representative": [r.get("average_recall_representative") for r in results_summary],
            "pyflat": [r.get("average_recall_python_flat") for r in results_summary],
            "milvus": [r.get("average_recall_milvus") for r in results_summary],
            "milvus_mix": [r.get("average_recall_milvus_mixed") for r in results_summary],
            "qdrant": [r.get("average_recall_qdrant") for r in results_summary],
            "qdrant_mix": [r.get("average_recall_qdrant_mixed") for r in results_summary],
            "chroma": [r.get("average_recall_chroma") for r in results_summary],
            "chroma_mix": [r.get("average_recall_chroma_mixed") for r in results_summary],
        }
        def _ms(r, key):
            v = r.get(key)
            return (v * 1000) if v is not None else None
        latency_ms = {
            "hdmg": [_ms(r, "avg_mixed_time") for r in results_summary],
            "representative": [_ms(r, "avg_representative_time") for r in results_summary],
            "pyflat": [_ms(r, "avg_python_flat_time") for r in results_summary],
            "milvus": [_ms(r, "avg_milvus_time") for r in results_summary],
            "milvus_mix": [_ms(r, "avg_milvus_mixed_time") for r in results_summary],
            "qdrant": [_ms(r, "avg_qdrant_time") for r in results_summary],
            "qdrant_mix": [_ms(r, "avg_qdrant_mixed_time") for r in results_summary],
            "chroma": [_ms(r, "avg_chroma_time") for r in results_summary],
            "chroma_mix": [_ms(r, "avg_chroma_mixed_time") for r in results_summary],
        }
        summary = {
            "bench": "yahoo",
            "expansion_factor": EXPANSION_FACTOR,
            "alpha": ALPHA,
            "betas": betas,
            "recall": recall,
            "latency_ms": latency_ms,
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary written to {args.summary_json}", flush=True)
    total_elapsed = time.time() - main_start
    print(f"\n=== Total time taken for this run: {total_elapsed:.2f} ({total_elapsed/60:.2f} ) ===")
