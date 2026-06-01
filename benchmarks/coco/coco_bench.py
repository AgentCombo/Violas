"""
COCO锛圝SON list + image path) semantic recall benchmark. Bucketed by COCO-80 categories predicted by CLIP, process aligned flower_bench.
"""
import argparse
import os
import sys
import time
import json
import pickle
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from coco_feature_extract import extract_all_features_from_coco_json, KeyPredictor
from violas.storage import VectorMap
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
    search_chroma_mixed
)
from violas.core.bench_recall_utils import (
    is_beta_pure_key,
    key_hit_slots_for_json,
    norm_key_for_recall,
    padded_norm_keys_from_search_results,
    padded_norm_keys_from_tuples,
    recall_key_hit_rate_at_k,
)
from violas.core.legacy_pickle import enable_legacy_storage_pickle

# Global variable: stores KeyPredictor instance for use by test_recall
_predictor = None
_python_flat_index = None
_python_flat_mode = "naive"


def build_vector_database_for_test(
    root,
    json_file="coco_dataset_40.json",
    max_items=None,
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
):
    """Build vector map table and display storage statistics
    Parameters:
        root: Data set root directory (including JSON and images)
        json_file: JSON file name relative to root
        max_items: Maximum number of items processed (default all)
        test_size: test set ratio, default 0.1 (10%锛?        random_state: Random seed to ensure reproducibility
        build_milvus: whether to build Milvus collection
        use_milvus_lite: whether to use Milvus Lite (embedded, no server required)
        milvus_host: Milvus server address (only if use_milvus_lite=False when used)
        milvus_port: Milvus server port (only if use_milvus_lite=False when used)
        build_qdrant: whether to build Qdrant collection
        build_chroma: whether to build Chroma collection
        use_cache: whether to enable local cache (cache directory)
        cache_prefix: cache prefix (used to distinguish data sets), automatically generated based on root by default

    Return:
        vectormap: vector mapping table (use 90%Data construction, key predicted by KeyPredictor)
        test_data: test data dictionary {folder_name: {'vectors': [...], 'descriptions': [...]}}
        milvus_collection: Milvus Collection Object (if built)
        milvus_id_map: Milvus ID mapping dictionary (if built)
        qdrant_collection: Qdrant Collection object (if built)
        chroma_collection: Chroma Collection object (if built)
    """
    global _predictor, _python_flat_index, _python_flat_mode

    print("=== Start building VectorMap and Milvus Collection===")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache_dir = os.path.join(project_root, "cache")
    legacy_cache_dir = os.path.abspath(os.path.join(project_root, "..", "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    if cache_prefix is None:
        dataset_name = os.path.basename(os.path.normpath(root))
        cache_prefix = re.sub(r"[^0-9a-zA-Z._-]+", "_", dataset_name)
    mi = f"mi{max_items}" if max_items is not None else "all"
    cache_name = f"coco_{cache_prefix}_{json_file}_{mi}_ts{test_size}_rs{random_state}_a{alpha:.3f}_folder.pkl"
    cache_file = os.path.join(cache_dir, cache_name)
    legacy_cache_file = os.path.join(legacy_cache_dir, cache_name)
    if not os.path.exists(cache_file) and os.path.exists(legacy_cache_file):
        cache_file = legacy_cache_file

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
        _predictor = KeyPredictor.from_key_lists(folders)
        print(f"[Cache] VectorMap/TestData loaded, number of keys: {len(_predictor.get_all_keys())}")
    else:
        print("\nStep 1: Extract CLIP features from COCO JSON and bucket by predicted class...")
        all_vectors, all_descriptions = extract_all_features_from_coco_json(
            root, json_file=json_file, max_items=max_items
        )
        folders = sorted(all_vectors.keys())
        print(f"Collected in total {len(folders)} folder data")

        # Step 2: Create KeyPredictor (extract all possible keys from folder name)
        print("\nStep 2: Create KeyPredictor...")
        _predictor = KeyPredictor.from_key_lists(folders)
        print(f"KeyPredictorCreated, supported {len(_predictor.get_all_keys())} key")

        # Step 3: Split train and test data
        print("\nStep 3: Split train and test data...")
        train_vectors_by_folder = {}  # {folder_name: [vector1, vector2, ...]}
        train_descriptions_by_folder = {}  # {folder_name: [desc1, desc2, ...]}
        test_data = {}  # {folder_name: {'vectors': [...], 'descriptions': [...]}}

        for folder in folders:
            if folder not in all_vectors:
                continue

            vectors = all_vectors[folder]
            descriptions = all_descriptions[folder]

            if len(vectors) == 0:
                continue

            # Small sample buckets (such as coco_dataset_40 used for smoke) may only have one picture to avoid the training set being empty after splitting.
            if len(vectors) < 2:
                train_vectors = list(vectors)
                test_vectors = []
                train_descriptions = list(descriptions)
                test_descriptions = []
            else:
                train_vectors, test_vectors, train_descriptions, test_descriptions = train_test_split(
                    vectors, descriptions,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=True
                )

            # Store test data (organized by folder)
            # train_test_split returns a numpy array, converted to a list
            test_data[folder] = {
                'vectors': [vec for vec in test_vectors],
                'descriptions': test_descriptions
            }

            # Store training data (organized by folder)
            train_vectors_by_folder[folder] = [vec for vec in train_vectors]
            train_descriptions_by_folder[folder] = train_descriptions

        print(f"Train/TestData splitting completed")
        print(f"The test set data has been stored separately, with a total of {len(test_data)} folders")

        # Step 4: Build VectorMap (including predict key and storage)
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
                }, f)
            print(f"[Cache] Written to cache: {cache_file}")

    # Step 5: Build Milvus collection (built directly from train_data_by_key)
    milvus_collection = None
    milvus_id_map = None
    if build_milvus:
        milvus_collection, milvus_id_map = build_milvus_collection_from_train_data(
            train_data_by_key,
            use_lite=use_milvus_lite,
            host=milvus_host,
            port=milvus_port
        )
    # Step 5.1: Build Qdrant
    qdrant_client, qdrant_id_map = None, None
    if build_qdrant:
        qdrant_client, qdrant_id_map = build_qdrant_collection_from_train_data(train_data_by_key)

    # Step 5.2: Build Chroma
    chroma_collection, chroma_id_map = None, None
    if build_chroma:
        chroma_collection, chroma_id_map = build_chroma_collection_from_train_data(train_data_by_key)

    # Step 6: Build a Python Flat prototype index (for comparison with Milvus)
    _python_flat_index = build_python_flat_index_from_train_data(train_data_by_key)
    if _python_flat_index is not None:
        print(f"PythonFlat Index built, number of vectors: {len(_python_flat_index['keys'])}", flush=True)

    return vectormap, test_data, milvus_collection, milvus_id_map, qdrant_client, qdrant_id_map, chroma_collection, chroma_id_map


def test_recall(vectormap: VectorMap, test_data, top_k=10, milvus_collection=None, milvus_id_map=None, qdrant_client=None, qdrant_id_map=None, chroma_collection=None, chroma_id_map=None,save_results=True, output_dir="outputs/coco", expansion_factor=10, beta=0.5, file_prefix=None, alpha=None, use_hdmg=False, hdmg_cluster_pool_size=64, hdmg_extra_hops=0, max_queries=None):
    """Test recall metrics

    Parameters:
        vectormap: vector mapping table
        test_data: test data dictionary
        top_k: Number of top k results searched, default 10
        milvus_collection: Milvus Collection object (optional)
        milvus_id_map: Milvus ID mapping dictionary (optional)
        qdrant_client: Qdrant Client object (optional)
        qdrant_id_map: Qdrant ID mapping dictionary (optional)
        chroma_collection: Chroma Collection object (optional)
        chroma_id_map: Chroma ID mapping dictionary (optional)
        save_results: whether to save detailed results to a file
        output_dir: directory where results are saved
        expansion_factor: group expansion factor (use_hdmg=False effective when)
        beta: weight of hybrid search (0.0: pure rep, 1.0: pure key)
        file_prefix: file name prefix (optional, timestamp will be automatically generated if not provided)
        alpha: grouping parameter (for recording in the result file)
        use_hdmg: If True, Mixed uses HDMG image retrieval (need to build_hdmg first), otherwise use search_with_mixed_key_rep_vec
        hdmg_cluster_pool_size: HDMG candidate micro-cluster pool size (use_hdmg=True effective when)
        max_queries: Maximum number of test queries, None or 0 means all

    Return:
        dict: Dictionary containing various statistical indicators
    """
    global _predictor, _python_flat_index
    print(f"\n{'='*80}")
    print(f"=== Recall test ===")
    print(f"Ground Truth: brute-force mixed (beta={beta}) Search top {top_k}")
    print(f"Mixed (beta={beta}{', HDMG' if use_hdmg else ''}): Mixed key+rep vector search top {top_k}")
    print(f"Representative: rep Check top-{top_k} Group, all vectors in the group use mixed scores to rerank top {top_k}")
    if _python_flat_index is not None:
        print(f"PythonFlat({ _python_flat_mode }): Flat + Cosine Search top {top_k}")
    if milvus_collection is not None:
        print(f"Milvus: vector search top {top_k}")
    print(f"{'='*80}\n")
    print(f"=== Parameter settings ===")
    print(f"alpha: {ALPHA}")
    print(f"expansion_factor: {EXPANSION_FACTOR}")
    print(f"beta: {BETA}")
    print(f"{'='*80}\n")

    def get_result_id(result):
        """Get the unique identifier of the search result (using img_path)"""
        if result.vector_idx is not None and result.vector_idx < len(result.group.descriptions):
            return result.group.descriptions[result.vector_idx].get('img_path')
        return None

    # Collect all test vectors
    all_test_samples = []
    for folder, data in test_data.items():
        vectors = data['vectors']
        descriptions = data['descriptions']
        for vec, desc in zip(vectors, descriptions):
            all_test_samples.append({
                'folder': folder,
                'vector': vec,
                'description': desc
            })

    if len(all_test_samples) == 0:
        print("warning: There is no data in the test set and recall testing cannot be performed.")
        return 0.0

    if max_queries is not None and max_queries > 0:
        all_test_samples = all_test_samples[:max_queries]
        print(f"Only before testing {len(all_test_samples)} (max_queries={max_queries})\n")
    else:
        print(f"test all {len(all_test_samples)} query\n")

    # Test mixed mode (ground truth is key-single)
    recalls_mixed = []  # mixed mode recall
    recalls_representative = []  # Representative mode recall (rep group search + mixed score rerank)
    recalls_python_flat = []  # Python Flat recall
    recalls_milvus = []  # Milvus' recall
    recalls_milvus_mixed = []  # Milvus hybrid search recall
    recalls_qdrant = [] # Qdrant's recall
    recalls_qdrant_mixed = [] # Qdrant mixed recall
    recalls_chroma = []   # Chroma's recall
    recalls_chroma_mixed = [] # Chroma mixed recall

    total_queries = 0
    successful_queries = 0

    # time statistics
    total_gt_time = 0.0  # Ground truth (key-single) total time
    total_mixed_time = 0.0  # Mixed total search time
    total_representative_time = 0.0  # Representative search total time
    total_python_flat_time = 0.0  # Python Flat search total time
    total_milvus_time = 0.0  # Milvus search total time
    total_milvus_mixed_time = 0.0  # Milvus hybrid search total time
    total_qdrant_time = 0.0 # Qdrant search total time
    total_qdrant_mixed_time = 0.0 # Qdrant mixed total search time
    total_chroma_time = 0.0   # Chroma search total time
    total_chroma_mixed_time = 0.0   # Chroma mixed total search time
    hdmg_route_depths = []
    hdmg_route_hops = []
    hdmg_candidate_pool_sizes = []
    hdmg_selected_nodes_sizes = []
    hdmg_mixed_score_calls_list = []
    hdmg_t_entry_list = []
    hdmg_t_walk_list = []
    hdmg_t_rerank_list = []

    def get_result_with_img_path(result):
        """Get search results and their img_path"""
        img_path = get_result_id(result)  # get_result_id now returns img_path directly
        # Also returns the original (key, group_name, vector_idx) for detailed results
        vector_idx = result.vector_idx if result.vector_idx is not None else -1
        result_id = (result.key, result.group.group_name, vector_idx)
        return result_id, img_path



    # Save detailed results
    detailed_results = []

    for idx, sample in enumerate(tqdm(all_test_samples, desc="Under test"), 1):
        query_vector = sample['vector']
        query_folder = sample['folder']

        try:
            # Directly use the folder name as the key
            predicted_key = query_folder

            # 1. Ground truth: brute force search using mixed_score with the same beta
            gt_start = time.time()
            query_key_vector = _predictor.get_key_vector(query_folder)
            gt_mixed_results = search_python_flat_mixed(
                _python_flat_index, query_vector, query_key_vector,
                vectormap.key_vectors, beta=beta, top_k=top_k
            )
            gt_end = time.time()
            total_gt_time += (gt_end - gt_start)

            gt_img_paths = [r[0] for r in gt_mixed_results]

            # 2. mixed mode: HDMG image retrieval or search_with_mixed_key_rep_vec
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
            hdmg_stats = None
            if use_hdmg and hasattr(vectormap, "get_last_hdmg_search_stats"):
                hdmg_stats = vectormap.get_last_hdmg_search_stats()
                if hdmg_stats and hdmg_stats.get("mode") == "hdmg":
                    hdmg_route_depths.extend(hdmg_stats.get("route_depths", []))
                    hdmg_route_hops.extend(hdmg_stats.get("route_hops", []))
                    hdmg_candidate_pool_sizes.append(hdmg_stats.get("candidate_pool_size", 0))
                    hdmg_selected_nodes_sizes.append(hdmg_stats.get("selected_nodes", 0))
                    hdmg_mixed_score_calls_list.append(hdmg_stats.get("mixed_score_calls", 0))
                    hdmg_t_entry_list.append(hdmg_stats.get("t_entry_ms", 0))
                    hdmg_t_walk_list.append(hdmg_stats.get("t_walk_ms", 0))
                    hdmg_t_rerank_list.append(hdmg_stats.get("t_rerank_ms", 0))

            # 3. Representative mode: rep checks the top-k groups, and reranks all vectors in the group using mixed scores.
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

            # 4. Python Flat prototype search results and time
            python_flat_results = None
            if _python_flat_index is not None:
                python_start = time.time()
                python_flat_results = search_python_flat_index(
                    _python_flat_index,
                    query_vector,
                    top_k=top_k,
                    mode=_python_flat_mode,
                )
                python_end = time.time()
                total_python_flat_time += (python_end - python_start)

            # 5.1 Milvus search results and time
            milvus_results = None
            if milvus_collection is not None and milvus_id_map is not None:
                milvus_start = time.time()
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
                milvus_search_results = milvus_collection.search(
                    [query_vec],
                    "vector",
                    search_params,
                    limit=top_k,
                    output_fields=["id"]
                )
                milvus_end = time.time()
                total_milvus_time += (milvus_end - milvus_start)

                # Convert Milvus results to our format
                if len(milvus_search_results) > 0:
                    milvus_ids = []  # Store (img_path, key) tuple
                    for hit in milvus_search_results[0]:
                        milvus_id = hit.id
                        if milvus_id in milvus_id_map:
                            milvus_ids.append(milvus_id_map[milvus_id])  # (img_path, key)
                    milvus_results = milvus_ids  # Keep it as a list, and extract img_path and key later.

            # 5.2 Milvus mixed: candidate pool + local mixed rerank (schema must contain key_str)
            milvus_mixed_results = None
            milvus_mixed_start = milvus_mixed_end = None
            if milvus_collection is not None and milvus_id_map is not None:
                milvus_mixed_start = time.time()
                milvus_mixed_results = search_milvus_mixed(
                    milvus_collection,
                    query_vector,
                    query_key_vector=query_key_vector,
                    key_vectors=vectormap.key_vectors,
                    id_to_result=milvus_id_map,
                    beta=beta,
                    top_k=top_k,
                    candidate_factor=10,
                )
                milvus_mixed_end = time.time()
                if milvus_mixed_results is not None:
                    total_milvus_mixed_time += (milvus_mixed_end - milvus_mixed_start)

            # 6. Qdrant search results
            qdrant_results = None
            qdrant_mixed_results = None
            if qdrant_client is not None and qdrant_id_map is not None:
                q_start = time.time()
                query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
                q_res = qdrant_client.query_points(
                    collection_name="flower-102",
                    query=query_vec,
                    limit=top_k
                )
                q_hits = q_res.points   # Extract the hit point list from the returned result
                q_end = time.time()
                total_qdrant_time += (q_end - q_start)
                qdrant_results = [qdrant_id_map[hit.id] for hit in q_hits if hit.id in qdrant_id_map]

                # Qdrant mixed
                qdrant_mixed_start = time.time()
                qdrant_mixed_results = search_qdrant_mixed(
                    qdrant_client,
                    "flower-102",
                    query_vector,
                    query_key_vector=query_key_vector,
                    key_vectors=vectormap.key_vectors,
                    id_to_result=qdrant_id_map,
                    beta=beta,
                    top_k=top_k,
                    candidate_factor=10,
                )
                qdrant_mixed_end = time.time()
                if qdrant_mixed_results is not None:
                    total_qdrant_mixed_time += (qdrant_mixed_end - qdrant_mixed_start)

            # 7. Chroma search logic (if multiple processes share the persistence path, Collection does not exist may be reported, and Chroma will be skipped after catch here)
            chroma_results = None
            chroma_mixed_results = None
            if chroma_collection is not None and chroma_id_map is not None:
                try:
                    # Pure vector search
                    c_start = time.time()
                    chroma_results = search_chroma_mixed(
                        chroma_collection, query_vector, query_key_vector, vectormap.key_vectors,
                        chroma_id_map, beta=0.0, top_k=top_k, candidate_factor=1
                    )
                    c_end = time.time()
                    total_chroma_time += (time.time() - c_start)

                    cm_start = time.time()
                    chroma_mixed_results = search_chroma_mixed(
                        chroma_collection, query_vector, query_key_vector, vectormap.key_vectors,
                        chroma_id_map, beta=beta, top_k=top_k, candidate_factor=10
                    )
                    cm_end = time.time()
                    total_chroma_mixed_time += (time.time() - cm_start)
                except Exception as e:
                    if idx == 1 or (idx % 50) == 0:
                        print(f"(folder={query_folder}): {e}", flush=True)
                    chroma_results = None
                    chroma_mixed_results = None

            # Calculate recall: beta<1, use img_path to collect / |GT|; beta=1, use oracle GT key to collect, according to rank: hits@k / k
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
                mixed_ids = representative_ids = None
            else:
                gt_ids = set(p for p in gt_img_paths if p is not None)
                mixed_ids = set(id for id in (get_result_id(result) for result in mixed_results) if id is not None)
                representative_ids = set(id for id in (get_result_id(result) for result in representative_results) if id is not None)
                recall_mixed = len(gt_ids & mixed_ids) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_mixed.append(recall_mixed)
                recall_representative = len(gt_ids & representative_ids) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_representative.append(recall_representative)

            # Python Flat's recall (pure embedding, consistent with GT only when beta=0)
            recall_python_flat = None
            python_img_paths = None
            if python_flat_results is not None:
                if use_key_recall:
                    ord_pf = padded_norm_keys_from_tuples(python_flat_results, nk, top_k)
                    recall_python_flat = recall_key_hit_rate_at_k(gt_ids, ord_pf, top_k)
                else:
                    python_img_paths = set(item[0] for item in python_flat_results if item[0] is not None)
                    recall_python_flat = len(gt_ids & python_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_python_flat.append(recall_python_flat)

            # Milvus's recall (pure embedding, consistent with GT only when beta=0)
            recall_milvus = None
            milvus_img_paths = None
            if milvus_results is not None:
                if use_key_recall:
                    ord_milvus = padded_norm_keys_from_tuples(milvus_results, nk, top_k)
                    recall_milvus = recall_key_hit_rate_at_k(gt_ids, ord_milvus, top_k)
                else:
                    milvus_img_paths = set(item[0] for item in milvus_results if item[0] is not None)
                    recall_milvus = len(gt_ids & milvus_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_milvus.append(recall_milvus)

            # Milvus mixed recall (candidate pool + mixed rerank)
            recall_milvus_mixed = None
            milvus_mixed_img_paths = set()
            if milvus_mixed_results is not None:
                if use_key_recall:
                    ord_milvus_m = padded_norm_keys_from_tuples(milvus_mixed_results, nk, top_k)
                    recall_milvus_mixed = recall_key_hit_rate_at_k(gt_ids, ord_milvus_m, top_k)
                else:
                    milvus_mixed_img_paths = set(item[0] for item in milvus_mixed_results if item[0] is not None)
                    recall_milvus_mixed = len(gt_ids & milvus_mixed_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_milvus_mixed.append(recall_milvus_mixed)

            # Qdrant's recall
            recall_qdrant = None
            qdrant_img_paths = None
            if qdrant_results is not None:
                if use_key_recall:
                    ord_q = padded_norm_keys_from_tuples(qdrant_results, nk, top_k)
                    recall_qdrant = recall_key_hit_rate_at_k(gt_ids, ord_q, top_k)
                else:
                    qdrant_img_paths = set(item[0] for item in qdrant_results if item[0] is not None)
                    recall_qdrant = len(gt_ids & qdrant_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_qdrant.append(recall_qdrant)

            # Qdrant mixed recall
            recall_qdrant_mixed = None
            qdrant_mixed_img_paths = set()
            if qdrant_mixed_results is not None:
                if use_key_recall:
                    ord_qm = padded_norm_keys_from_tuples(qdrant_mixed_results, nk, top_k)
                    recall_qdrant_mixed = recall_key_hit_rate_at_k(gt_ids, ord_qm, top_k)
                else:
                    qdrant_mixed_img_paths = set(item[0] for item in qdrant_mixed_results if item[0] is not None)
                    recall_qdrant_mixed = len(gt_ids & qdrant_mixed_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_qdrant_mixed.append(recall_qdrant_mixed)
            # Chroma's recall
            recall_chroma = None
            chroma_img_paths = None
            if chroma_results is not None:
                if use_key_recall:
                    ord_c = padded_norm_keys_from_tuples(chroma_results, nk, top_k)
                    recall_chroma = recall_key_hit_rate_at_k(gt_ids, ord_c, top_k)
                else:
                    chroma_img_paths = set(item[0] for item in chroma_results if item[0] is not None)
                    recall_chroma = len(gt_ids & chroma_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_chroma.append(recall_chroma)

            # Chroma mixed recall
            recall_chroma_mixed = None
            chroma_mixed_img_paths = set()
            if chroma_mixed_results is not None:
                if use_key_recall:
                    ord_cm = padded_norm_keys_from_tuples(chroma_mixed_results, nk, top_k)
                    recall_chroma_mixed = recall_key_hit_rate_at_k(gt_ids, ord_cm, top_k)
                else:
                    chroma_mixed_img_paths = set(item[0] for item in chroma_mixed_results if item[0] is not None)
                    recall_chroma_mixed = len(gt_ids & chroma_mixed_img_paths) / len(gt_ids) if len(gt_ids) > 0 else 0.0
                recalls_chroma_mixed.append(recall_chroma_mixed)

            # Extract the img_path of each result
            def get_results_with_img_paths(results):
                """Get a list of results and their img_paths"""
                result_list = []
                for result in results:
                    result_id, img_path = get_result_with_img_path(result)
                    result_list.append({
                        'id': list(result_id),
                        'img_path': img_path,
                        'distance': float(result.distance)
                    })
                return result_list

            # Save detailed results
            query_result = {
                'query_id': idx,
                'query_folder': query_folder,
                'query_description': sample['description'],
                'predicted_key': predicted_key,
                'recall_mixed': recall_mixed,
                'recall_representative': recall_representative,
                'recall_python_flat': recall_python_flat,
                'recall_milvus': recall_milvus,
                'recall_milvus_mixed': recall_milvus_mixed,
                'recall_qdrant': recall_qdrant,
                'recall_qdrant_mixed': recall_qdrant_mixed,
                'recall_chroma': recall_chroma,
                'recall_chroma_mixed': recall_chroma_mixed,
                'ground_truth': [
                    {'img_path': r[0], 'key': r[1], 'distance': float(r[2])}
                    for r in gt_mixed_results
                ],
                'mixed_results': get_results_with_img_paths(mixed_results),
                'representative_results': get_results_with_img_paths(representative_results),
                'python_flat_results': [
                    {'img_path': item[0], 'key': item[1], 'distance': float(item[2])}
                    for item in python_flat_results
                ] if python_flat_results is not None else None,
                'python_flat_stats': None,
                'milvus_ids': [(item[0], item[1]) for item in milvus_results] if milvus_results is not None else None,
                'milvus_mixed_ids': [(item[0], item[1]) for item in milvus_mixed_results] if milvus_mixed_results is not None else None,
                'qdrant_ids': [(item[0], item[1]) for item in qdrant_results] if qdrant_results is not None else None,
                'qdrant_mixed_ids': [(item[0], item[1]) for item in qdrant_mixed_results] if qdrant_mixed_results is not None else None,
                'chroma_ids': [(item[0], item[1]) for item in chroma_results] if chroma_results is not None else None,
                'chroma_mixed_ids': [(item[0], item[1]) for item in chroma_mixed_results] if chroma_mixed_results is not None else None,
                'intersection_mixed': (
                    key_hit_slots_for_json(gt_ids, ord_mixed, top_k) if use_key_recall else list(gt_ids & mixed_ids)
                ),
                'intersection_representative': (
                    key_hit_slots_for_json(gt_ids, ord_rep, top_k) if use_key_recall else list(gt_ids & representative_ids)
                ),
                'intersection_python_flat': (
                    None
                    if python_flat_results is None
                    else (
                        key_hit_slots_for_json(gt_ids, ord_pf, top_k)
                        if use_key_recall
                        else list(gt_ids & python_img_paths)
                    )
                ),
                'intersection_milvus': (
                    None
                    if milvus_results is None
                    else (
                        key_hit_slots_for_json(gt_ids, ord_milvus, top_k)
                        if use_key_recall
                        else list(gt_ids & milvus_img_paths)
                    )
                ),
                'intersection_milvus_mixed': (
                    None
                    if milvus_mixed_results is None
                    else (
                        key_hit_slots_for_json(gt_ids, ord_milvus_m, top_k)
                        if use_key_recall
                        else list(gt_ids & milvus_mixed_img_paths)
                    )
                ),
                'gt_time': gt_end - gt_start,
                'mixed_time': mixed_end - mixed_start,
                'representative_time': representative_end - representative_start,
                'python_flat_time': python_end - python_start if python_flat_results is not None else None,
                'milvus_time': milvus_end - milvus_start if milvus_results is not None else None,
                'milvus_mixed_time': (milvus_mixed_end - milvus_mixed_start) if milvus_mixed_results is not None else None,
                'qdrant_time': q_end - q_start if qdrant_results is not None else None,
                'qdrant_mixed_time': (qdrant_mixed_end - qdrant_mixed_start) if qdrant_mixed_results is not None else None,
                'chroma_time': c_end - c_start if chroma_results is not None else None,
                'chroma_mixed_time': cm_end - cm_start if chroma_mixed_results is not None else None,
                'hdmg_stats': hdmg_stats if use_hdmg else None
            }
            detailed_results.append(query_result)

            total_queries += 1
            successful_queries += 1

        except Exception as e:
            print(f"\n(folder={query_folder}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            total_queries += 1
            continue

    # Calculate statistics
    if len(recalls_mixed) == 0:
        print("warning: No successful query, recall cannot be calculated")
        return 0.0

    # Mixed mode statistics
    average_recall_mixed = np.mean(recalls_mixed)
    min_recall_mixed = np.min(recalls_mixed)
    max_recall_mixed = np.max(recalls_mixed)

    # Statistics of representative pattern
    average_recall_representative = np.mean(recalls_representative)
    min_recall_representative = np.min(recalls_representative)
    max_recall_representative = np.max(recalls_representative)

    # Statistics for python_flat
    average_recall_python_flat = None
    if len(recalls_python_flat) > 0:
        average_recall_python_flat = np.mean(recalls_python_flat)

    # Milvus statistics
    average_recall_milvus = None
    if len(recalls_milvus) > 0:
        average_recall_milvus = np.mean(recalls_milvus)

    # Milvus mixed statistics
    average_recall_milvus_mixed = None
    if len(recalls_milvus_mixed) > 0:
        average_recall_milvus_mixed = np.mean(recalls_milvus_mixed)

    # Qdrant statistics
    average_recall_qdrant = None
    if len(recalls_qdrant) > 0:
        average_recall_qdrant = np.mean(recalls_qdrant)
    # Statistics for Qdrant mixed
    average_recall_qdrant_mixed = None
    if len(recalls_qdrant_mixed) > 0:
        average_recall_qdrant_mixed = np.mean(recalls_qdrant_mixed)
    # Chroma Statistics
    average_recall_chroma = None
    if len(recalls_chroma) > 0:
        average_recall_chroma = np.mean(recalls_chroma)
    # Chroma mixed statistics
    average_recall_chroma_mixed = None
    if len(recalls_chroma_mixed) > 0:
        average_recall_chroma_mixed = np.mean(recalls_chroma_mixed)
    # Calculate average time
    avg_gt_time = total_gt_time / successful_queries if successful_queries > 0 else 0.0
    avg_mixed_time = total_mixed_time / successful_queries if successful_queries > 0 else 0.0
    avg_representative_time = total_representative_time / successful_queries if successful_queries > 0 else 0.0
    avg_python_flat_time = total_python_flat_time / successful_queries if successful_queries > 0 and _python_flat_index is not None else 0.0
    avg_milvus_time = total_milvus_time / successful_queries if successful_queries > 0 and milvus_collection is not None else 0.0
    avg_milvus_mixed_time = total_milvus_mixed_time / successful_queries if successful_queries > 0 and milvus_mixed_results is not None else 0.0
    avg_qdrant_time = total_qdrant_time / successful_queries if successful_queries > 0 and qdrant_client is not None else 0.0
    avg_qdrant_mixed_time = total_qdrant_mixed_time / successful_queries if successful_queries > 0 and qdrant_client is not None else 0.0
    avg_chroma_time = total_chroma_time / successful_queries if successful_queries > 0 and chroma_collection is not None else 0.0
    avg_chroma_mixed_time = total_chroma_mixed_time / successful_queries if successful_queries > 0 and chroma_collection is not None else 0.0
    avg_hdmg_depth = np.mean(hdmg_route_depths) if len(hdmg_route_depths) > 0 else None
    avg_hdmg_hops = np.mean(hdmg_route_hops) if len(hdmg_route_hops) > 0 else None
    avg_hdmg_candidate_pool = np.mean(hdmg_candidate_pool_sizes) if len(hdmg_candidate_pool_sizes) > 0 else None
    avg_hdmg_selected_nodes = np.mean(hdmg_selected_nodes_sizes) if len(hdmg_selected_nodes_sizes) > 0 else None
    avg_hdmg_mixed_score_calls = np.mean(hdmg_mixed_score_calls_list) if len(hdmg_mixed_score_calls_list) > 0 else None

    # Print results
    print(f"\n{'='*80}")
    print(f"=== Recall Test results ===")
    print(f"{'='*80}")
    print(f"Total queries: {total_queries}")
    print(f"Number of successful queries: {successful_queries}")
    print(f"\n--- Recall Indicator header ---")
    print(f"| Method                | Avg Recall@{top_k} | Min Recall | Max Recall |")
    print(f"|-----------------------|-------------------|------------|-----------|")
    print(f"| Mixed (beta={beta}{', HDMG' if use_hdmg else ''}) | {average_recall_mixed:.4f}         | {min_recall_mixed:.4f}   | {max_recall_mixed:.4f}   |")
    print(f"| Representative        | {average_recall_representative:.4f}         | {min_recall_representative:.4f}   | {max_recall_representative:.4f}   |")
    if average_recall_python_flat is not None:
        print(f"| PythonFlat            | {average_recall_python_flat:.4f}         |      -     |     -     |")
    if average_recall_milvus is not None:
        print(f"| Milvus                | {average_recall_milvus:.4f}         |      -     |     -     |")
    if average_recall_milvus_mixed is not None:
        print(f"| Milvus Mixed          | {average_recall_milvus_mixed:.4f}         |      -     |     -     |")
    if average_recall_qdrant is not None:
        print(f"| Qdrant                | {average_recall_qdrant:.4f}         |      -     |     -     |")
    if average_recall_qdrant_mixed is not None:
        print(f"| Qdrant Mixed          | {average_recall_qdrant_mixed:.4f}         |      -     |     -     |")
    if average_recall_chroma is not None:
        print(f"| Chroma                | {average_recall_chroma:.4f}         |      -     |     -     |")
    if average_recall_chroma_mixed is not None:
        print(f"| Chroma Mixed          | {average_recall_chroma_mixed:.4f}         |      -     |     -     |")
    print(f"\n--- Time comparison ---")
    print(f"Ground Truth (brute-force mixed, beta={beta}) Total time: {total_gt_time:.2f} ({total_gt_time/60:.2f} )")
    print(f"Ground Truth Average query: {avg_gt_time*1000:.2f} millisecond")
    print(f"Mixed Total search time: {total_mixed_time:.2f} ({total_mixed_time/60:.2f} )")
    print(f"Mixed Average query: {avg_mixed_time*1000:.2f} millisecond")
    print(f"Representative Total search time: {total_representative_time:.2f} ({total_representative_time/60:.2f} )")
    print(f"Representative Average query: {avg_representative_time*1000:.2f} millisecond")
    if use_hdmg and avg_hdmg_depth is not None:
        print(f"HDMG (steps): {avg_hdmg_depth:.2f}")
        print(f"HDMG (hops): {avg_hdmg_hops:.2f}")
        print(f"HDMG Average candidate pool size: {avg_hdmg_candidate_pool:.2f}")
        print(f"HDMG Average final number of micro-clusters: {avg_hdmg_selected_nodes:.2f}")
        if avg_hdmg_mixed_score_calls is not None:
            print(f"HDMG Average number of mixed_score calls: {avg_hdmg_mixed_score_calls:.2f}")
        if len(hdmg_t_entry_list) > 0:
            print(f"HDMG Average time taken: entrance selection={np.mean(hdmg_t_entry_list):.3f}ms, "
                  f"Figure wandering={np.mean(hdmg_t_walk_list):.3f}ms, "
                  f"Reorder={np.mean(hdmg_t_rerank_list):.3f}ms")
    if _python_flat_index is not None:
        print(f"PythonFlat Total search time: {total_python_flat_time:.2f} ({total_python_flat_time/60:.2f} )")
        print(f"PythonFlat Average query: {avg_python_flat_time*1000:.2f} millisecond")
    if milvus_collection is not None:
        print(f"Milvus Total search time: {total_milvus_time:.2f} ({total_milvus_time/60:.2f} )")
        print(f"Milvus Average query: {avg_milvus_time*1000:.2f} millisecond")
    if milvus_mixed_results is not None:
        print(f"Milvus Mixed Total search time: {total_milvus_mixed_time:.2f} ({total_milvus_mixed_time/60:.2f} )")
        print(f"Milvus Mixed Average query: {avg_milvus_mixed_time*1000:.2f} millisecond")
    if qdrant_client is not None:
        print(f"Qdrant Total search time: {total_qdrant_time:.2f} ({total_qdrant_time/60:.2f} )")
        print(f"Qdrant Average query: {avg_qdrant_time*1000:.2f} millisecond")
        print(f"Qdrant Mixed Total search time: {total_qdrant_mixed_time:.2f} ({total_qdrant_mixed_time/60:.2f} )")
        print(f"Qdrant Mixed Average query: {avg_qdrant_mixed_time*1000:.2f} millisecond")
    if chroma_collection is not None:
        print(f"Chroma Total search time: {total_chroma_time:.2f} ({total_chroma_time/60:.2f} )")
        print(f"Chroma Average query: {avg_chroma_time*1000:.2f} millisecond")
        print(f"Chroma Mixed Total search time: {total_chroma_mixed_time:.2f} ({total_chroma_mixed_time/60:.2f} )")
        print(f"Chroma Mixed Average query: {avg_chroma_mixed_time*1000:.2f} millisecond")
    if total_gt_time > 0:
        speedup_mixed = total_gt_time / total_mixed_time if total_mixed_time > 0 else 0
        print(f"\nMixed Speedup ratio: {speedup_mixed:.2f}x (Faster than Ground Truth {speedup_mixed:.2f} )")
        speedup_representative = total_gt_time / total_representative_time if total_representative_time > 0 else 0
        print(f"Representative Speedup ratio: {speedup_representative:.2f}x (Faster than Ground Truth {speedup_representative:.2f} )")
        if _python_flat_index is not None and total_python_flat_time > 0:
            speedup_python_flat = total_gt_time / total_python_flat_time
            print(f"PythonFlat Speedup ratio: {speedup_python_flat:.2f}x (Faster than Ground Truth {speedup_python_flat:.2f} )")
        if milvus_collection is not None and total_milvus_time > 0:
            speedup_milvus = total_gt_time / total_milvus_time
            print(f"Milvus Speedup ratio: {speedup_milvus:.2f}x (Faster than Ground Truth {speedup_milvus:.2f} )")
        if qdrant_client is not None and total_qdrant_time > 0:
            speedup_qdrant = total_gt_time / total_qdrant_time
            print(f"Qdrant Speedup ratio: {speedup_qdrant:.2f}x (Faster than Ground Truth {speedup_qdrant:.2f} )")
        if qdrant_client is not None and total_qdrant_mixed_time > 0:
            speedup_qdrant_mixed = total_gt_time / total_qdrant_mixed_time
            print(f"Qdrant Mixed Speedup ratio: {speedup_qdrant_mixed:.2f}x (Faster than Ground Truth {speedup_qdrant_mixed:.2f} )")
        if chroma_collection is not None and total_chroma_time > 0:
            speedup_chroma = total_gt_time / total_chroma_time
            print(f"Chroma Speedup ratio: {speedup_chroma:.2f}x (Faster than Ground Truth {speedup_chroma:.2f} )")
        if chroma_collection is not None and total_chroma_mixed_time > 0:
            speedup_chroma_mixed = total_gt_time / total_chroma_mixed_time
            print(f"Chroma Mixed Speedup ratio: {speedup_chroma_mixed:.2f}x (Faster than Ground Truth {speedup_chroma_mixed:.2f} )")
        print(f"{'='*80}\n")

    # Save detailed results to file
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

        # Use the passed in prefix or automatically generate a timestamp
        if file_prefix is None:
            file_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        # The file name contains the parameter value: prefix_exp{expansion_factor}_beta{beta}.json
        filename = f"recall_detailed_{file_prefix}_exp{expansion_factor}_beta{beta:.1f}.json"
        detailed_file = os.path.join(output_dir, filename)

        summary = {
            'test_config': {
                'top_k': top_k,
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'has_python_flat': _python_flat_index is not None,
                'has_milvus': milvus_collection is not None,
                'has_qdrant': qdrant_client is not None,
                'has_chroma': chroma_collection is not None,
                'ground_truth_method': 'key-single',  # Mark the method used by ground truth
                'alpha': alpha,  # Grouping parameters
                'beta': beta,  # beta parameters for hybrid search
                'expansion_factor': expansion_factor,
                'use_hdmg': use_hdmg,
                'file_prefix': file_prefix
            },
            'statistics': {
                'average_recall_mixed': float(average_recall_mixed),
                'average_recall_representative': float(average_recall_representative),
                'average_recall_python_flat': float(average_recall_python_flat) if average_recall_python_flat is not None else None,
                'average_recall_milvus': float(average_recall_milvus) if average_recall_milvus is not None else None,
                'average_recall_milvus_mixed': float(average_recall_milvus_mixed) if average_recall_milvus_mixed is not None else None,
                'average_recall_qdrant': float(average_recall_qdrant) if average_recall_qdrant is not None else None,
                'average_recall_qdrant_mixed': float(average_recall_qdrant_mixed) if average_recall_qdrant_mixed is not None else None,
                'average_recall_chroma': float(average_recall_chroma) if average_recall_chroma is not None else None,
                'average_recall_chroma_mixed': float(average_recall_chroma_mixed) if average_recall_chroma_mixed is not None else None,
                'min_recall_mixed': float(min_recall_mixed),
                'max_recall_mixed': float(max_recall_mixed),
                'min_recall_representative': float(min_recall_representative),
                'max_recall_representative': float(max_recall_representative),
                'total_gt_time': float(total_gt_time),
                'total_mixed_time': float(total_mixed_time),
                'total_representative_time': float(total_representative_time),
                'total_python_flat_time': float(total_python_flat_time) if _python_flat_index is not None else None,
                'total_milvus_time': float(total_milvus_time) if milvus_collection is not None else None,
                'total_milvus_mixed_time': float(total_milvus_mixed_time) if milvus_mixed_results is not None else None,
                'total_qdrant_time': float(total_qdrant_time) if qdrant_client is not None else None,
                'total_qdrant_mixed_time': float(total_qdrant_mixed_time) if qdrant_client is not None else None,
                'total_chroma_time': float(total_chroma_time) if chroma_collection is not None else None,
                'total_chroma_mixed_time': float(total_chroma_mixed_time) if chroma_collection is not None else None,
                'avg_gt_time': float(avg_gt_time),
                'avg_mixed_time': float(avg_mixed_time),
                'avg_representative_time': float(avg_representative_time),
                'avg_python_flat_time': float(avg_python_flat_time) if _python_flat_index is not None else None,
                'avg_milvus_time': float(avg_milvus_time) if milvus_collection is not None else None,
                'avg_milvus_mixed_time': float(avg_milvus_mixed_time) if milvus_mixed_results is not None else None,
                'avg_qdrant_time': float(avg_qdrant_time) if qdrant_client is not None else None,
                'avg_qdrant_mixed_time': float(avg_qdrant_mixed_time) if qdrant_client is not None else None,
                'avg_chroma_time': float(avg_chroma_time) if chroma_collection is not None else None,
                'avg_chroma_mixed_time': float(avg_chroma_mixed_time) if chroma_collection is not None else None
                },
            'detailed_results': detailed_results
        }

        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Detailed results have been saved to: {detailed_file}")

    # Returns the complete statistics dictionary
    return {
        'average_recall_mixed': float(average_recall_mixed),
        'average_recall_representative': float(average_recall_representative),
        'average_recall_python_flat': float(average_recall_python_flat) if average_recall_python_flat is not None else None,
        'average_recall_milvus': float(average_recall_milvus) if average_recall_milvus is not None else None,
        'average_recall_milvus_mixed': float(average_recall_milvus_mixed) if average_recall_milvus_mixed is not None else None,
        'average_recall_qdrant': float(average_recall_qdrant) if average_recall_qdrant is not None else None,
        'average_recall_qdrant_mixed': float(average_recall_qdrant_mixed) if average_recall_qdrant_mixed is not None else None,
        'average_recall_chroma': float(average_recall_chroma) if average_recall_chroma is not None else None,
        'average_recall_chroma_mixed': float(average_recall_chroma_mixed) if average_recall_chroma_mixed is not None else None,
        'min_recall_mixed': float(min_recall_mixed),
        'max_recall_mixed': float(max_recall_mixed),
        'min_recall_representative': float(min_recall_representative),
        'max_recall_representative': float(max_recall_representative),
        'total_gt_time': float(total_gt_time),
        'total_mixed_time': float(total_mixed_time),
        'total_representative_time': float(total_representative_time),
        'total_python_flat_time': float(total_python_flat_time) if _python_flat_index is not None else None,
        'total_milvus_time': float(total_milvus_time) if milvus_collection is not None else None,
        'total_milvus_mixed_time': float(total_milvus_mixed_time) if milvus_mixed_results is not None else None,
        'total_qdrant_time': float(total_qdrant_time) if qdrant_client is not None else None,
        'total_qdrant_mixed_time': float(total_qdrant_mixed_time) if qdrant_client is not None else None,
        'total_chroma_time': float(total_chroma_time) if chroma_collection is not None else None,
        'total_chroma_mixed_time': float(total_chroma_mixed_time) if chroma_collection is not None else None,
        'avg_gt_time': float(avg_gt_time),
        'avg_mixed_time': float(avg_mixed_time),
        'avg_representative_time': float(avg_representative_time),
        'avg_python_flat_time': float(avg_python_flat_time) if _python_flat_index is not None else None,
        'avg_milvus_time': float(avg_milvus_time) if milvus_collection is not None else None,
        'avg_milvus_mixed_time': float(avg_milvus_mixed_time) if milvus_mixed_results is not None else None,
        'avg_qdrant_time': float(avg_qdrant_time) if qdrant_client is not None else None,
        'avg_qdrant_mixed_time': float(avg_qdrant_mixed_time) if qdrant_client is not None else None,
        'avg_chroma_time': float(avg_chroma_time) if chroma_collection is not None else None,
        'avg_chroma_mixed_time': float(avg_chroma_mixed_time) if chroma_collection is not None else None,
        'hdmg_avg_depth': float(avg_hdmg_depth) if avg_hdmg_depth is not None else None,
        'hdmg_avg_hops': float(avg_hdmg_hops) if avg_hdmg_hops is not None else None,
        'hdmg_avg_candidate_pool': float(avg_hdmg_candidate_pool) if avg_hdmg_candidate_pool is not None else None,
        'hdmg_avg_mixed_score_calls': float(avg_hdmg_mixed_score_calls) if avg_hdmg_mixed_score_calls is not None else None,
        'hdmg_avg_t_entry_ms': float(np.mean(hdmg_t_entry_list)) if len(hdmg_t_entry_list) > 0 else None,
        'hdmg_avg_t_walk_ms': float(np.mean(hdmg_t_walk_list)) if len(hdmg_t_walk_list) > 0 else None,
        'hdmg_avg_t_rerank_ms': float(np.mean(hdmg_t_rerank_list)) if len(hdmg_t_rerank_list) > 0 else None,
    }


if __name__ == "__main__":
    _default_root = os.path.abspath(os.path.join(REPO_ROOT, "..", "dataset", "coco"))

    parser = argparse.ArgumentParser(description="COCO JSON recall benchmark (CLIP + COCO-80 buckets)")
    parser.add_argument("--root", default=_default_root, help="Dataset root path (including JSON and images)")
    parser.add_argument("--json_file", default="coco_dataset_40.json", help="Annotation JSON relative to root")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of JSON items processed (debugging)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved items")
    parser.add_argument("--alpha", type=float, default=0.5, help="Within-group clustering strength 0~1")
    parser.add_argument("--beta", type=float, default=0.3, help="Mixed search key weight 0~1")
    parser.add_argument("--expansion_factor", type=int, default=3, help="candidate pool multiple")
    parser.add_argument("--hdmg_embedding_k", type=int, default=12)
    parser.add_argument("--hdmg_semantic_intra_k", type=int, default=20)
    parser.add_argument("--hdmg_semantic_bridge_keys", type=int, default=2)
    parser.add_argument("--hdmg_semantic_bridge_per_key", type=int, default=1)
    parser.add_argument("--hdmg_cluster_pool_size", type=int, default=9, help="Turning up the recall is usually better")
    parser.add_argument("--hdmg_extra_hops", type=int, default=0)
    parser.add_argument("--python_flat_mode", default="naive", choices=("naive", "vectorized"))
    parser.add_argument("--max_queries", type=int, default=200, help="Maximum number of test queries, 0 means all")
    parser.add_argument("--summary_json", default=None, help="If specified, write the parameter test summary and time comparison to the JSON file")
    parser.add_argument("--only_beta", type=float, default=None, help="If specified, only this beta will be run (overriding the default beta scan)")
    args = parser.parse_args()

    root = args.root
    enable_external_dbs = os.environ.get("VIOLAS_ENABLE_EXTERNAL_DBS", "0") == "1"
    BUILD_MILVUS = enable_external_dbs
    USE_MILVUS_LITE = True
    BUILD_QDRANT = enable_external_dbs
    BUILD_CHROMA = enable_external_dbs
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"

    TOP_K = args.top_k
    ALPHA = args.alpha
    BETA = args.beta
    EXPANSION_FACTOR = args.expansion_factor
    HDMG_EMBEDDING_K = args.hdmg_embedding_k
    HDMG_SEMANTIC_INTRA_K = args.hdmg_semantic_intra_k
    HDMG_SEMANTIC_BRIDGE_KEYS = args.hdmg_semantic_bridge_keys
    HDMG_SEMANTIC_BRIDGE_PER_KEY = args.hdmg_semantic_bridge_per_key
    HDMG_CLUSTER_POOL_SIZE = args.hdmg_cluster_pool_size
    HDMG_EXTRA_HOPS = args.hdmg_extra_hops
    PYTHON_FLAT_MODE = args.python_flat_mode
    MAX_QUERIES = args.max_queries if args.max_queries > 0 else None

    _python_flat_mode = PYTHON_FLAT_MODE

    main_start_time = time.time()

    # Build vector mapping table (90% data) and test set (10% data)
    build_start_time = time.time()
    vectormap, test_data, milvus_collection, milvus_id_map, qdrant_client, qdrant_id_map, chroma_collection, chroma_id_map = build_vector_database_for_test(
        root,
        json_file=args.json_file,
        max_items=args.max_items,
        test_size=0.1,
        build_milvus=BUILD_MILVUS,
        use_milvus_lite=USE_MILVUS_LITE,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        alpha=ALPHA,
        build_qdrant=BUILD_QDRANT,
        build_chroma=BUILD_CHROMA,
    )
    build_end_time = time.time()
    build_duration = build_end_time - build_start_time
    print(f"VectorMapAnd the construction of Milvus Collection is completed, which takes: {build_duration:.2f} ({build_duration/60:.2f} )", flush=True)

    # Build HDMG graph index (for Mixed acceleration)
    hdmg_start = time.time()
    vectormap.build_hdmg(
        embedding_k=HDMG_EMBEDDING_K,
        semantic_intra_k=HDMG_SEMANTIC_INTRA_K,
        semantic_bridge_keys=HDMG_SEMANTIC_BRIDGE_KEYS,
        semantic_bridge_per_key=HDMG_SEMANTIC_BRIDGE_PER_KEY,
        use_mutual_embedding=False,
    )
    print(f"HDMG The graph index construction is completed and takes: {time.time() - hdmg_start:.2f} Second", flush=True)

    # Print experimental parameters
    print(f"\n=== Experimental parameters ===")
    print(f"TOP_K={TOP_K}, ALPHA={ALPHA}, BETA={BETA}, EXPANSION_FACTOR={EXPANSION_FACTOR}")
    print(f"HDMG: embedding_k={HDMG_EMBEDDING_K}, semantic_intra_k={HDMG_SEMANTIC_INTRA_K}, "
          f"bridge_keys={HDMG_SEMANTIC_BRIDGE_KEYS}, bridge_per_key={HDMG_SEMANTIC_BRIDGE_PER_KEY}, "
          f"cluster_pool_size={HDMG_CLUSTER_POOL_SIZE}, extra_hops={HDMG_EXTRA_HOPS}")
    print(f"PYTHON_FLAT_MODE={PYTHON_FLAT_MODE}")

    # Print test set statistics
    print(f"\n=== Test set statistics ===")
    vectormap.analyze_vectormap_storage()
    total_test_vectors = sum(len(data['vectors']) for data in test_data.values())
    print(f"Number of test set folders: {len(test_data)}")
    print(f"Total number of vectors in test set: {total_test_vectors}")

    # Recall test - multiple sets of beta parameters (consistent with caltech/run_experiments.sh)
    param_combinations = (
        [args.only_beta]
        if args.only_beta is not None
        else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    print(f"\n{'='*80}")
    print(f"=== Multiple sets of parameters Recall test ===")
    print(f"Testing {len(param_combinations)} parameter combinations")
    print(f"alpha: {ALPHA}")
    print(f"hdmg_cluster_pool_size: {HDMG_CLUSTER_POOL_SIZE}")
    print(f"{'='*80}\n")

    # Generate a unified file prefix (common to all tests)
    test_run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"This test file prefix: {test_run_prefix}")

    results_summary = []

    for i, beta in enumerate(param_combinations):
        print(f"\n--- No. {i+1}/{len(param_combinations)} Group parameters ---")
        print(f"beta: {beta}")

        query_start_time = time.time()
        stats = test_recall(
            vectormap,
            test_data,
            top_k=TOP_K,
            milvus_collection=milvus_collection,
            milvus_id_map=milvus_id_map,
            qdrant_client=qdrant_client,
            qdrant_id_map=qdrant_id_map,
            chroma_collection=chroma_collection,
            chroma_id_map=chroma_id_map,
            expansion_factor=EXPANSION_FACTOR,
            beta=beta,
            file_prefix=test_run_prefix,
            alpha=ALPHA,
            use_hdmg=True,
            hdmg_cluster_pool_size=HDMG_CLUSTER_POOL_SIZE,
            hdmg_extra_hops=HDMG_EXTRA_HOPS,
            max_queries=MAX_QUERIES,
        )
        query_end_time = time.time()
        query_duration = query_end_time - query_start_time

        results_summary.append({
            'expansion_factor': EXPANSION_FACTOR,
            'beta': beta,
            'duration': query_duration,
            **stats  # Expand all statistics
        })

        print(f"Recall The test is completed and takes: {query_duration:.2f} ({query_duration/60:.2f} )", flush=True)

    # Print summary results
    print(f"\n{'='*120}")
    print(f"=== (alpha={ALPHA}) ===")
    print(f"{'='*120}")

    # Header [Complete: add qdrant/chroma mixed]
    print(f"{'exp_factor':<12} {'beta':<6} | {'mixed_recall':<13} {'min':<6} {'max':<6} | {'representative':<12} {'min':<6} {'max':<6} | {'pyflat':<8} {'milvus':<8} {'milvus_mix':<10} {'qdrant':<8} {'qdrant_mix':<10} {'chroma':<8} {'chroma_mix':<12}")
    print("-" * 140)
    for r in results_summary:
        pyflat_recall = f"{r['average_recall_python_flat']:.4f}" if r['average_recall_python_flat'] is not None else "N/A"
        milvus_recall = f"{r['average_recall_milvus']:.4f}" if r['average_recall_milvus'] is not None else "N/A"
        milvus_mixed_recall = f"{r.get('average_recall_milvus_mixed'):.4f}" if r.get('average_recall_milvus_mixed') is not None else "N/A"
        qdrant_recall = f"{r.get('average_recall_qdrant'):.4f}" if r.get('average_recall_qdrant') is not None else "N/A"
        qdrant_mixed_recall = f"{r.get('average_recall_qdrant_mixed'):.4f}" if r.get('average_recall_qdrant_mixed') is not None else "N/A"
        chroma_recall = f"{r.get('average_recall_chroma'):.4f}" if r.get('average_recall_chroma') is not None else "N/A"
        chroma_mixed_recall = f"{r.get('average_recall_chroma_mixed'):.4f}" if r.get('average_recall_chroma_mixed') is not None else "N/A"
        print(f"{r['expansion_factor']:<12} {r['beta']:<6.2f} | "
              f"{r['average_recall_mixed']:<13.4f} {r['min_recall_mixed']:<6.2f} {r['max_recall_mixed']:<6.2f} | "
              f"{r['average_recall_representative']:<12.4f} {r['min_recall_representative']:<6.2f} {r['max_recall_representative']:<6.2f} | "
              f"{pyflat_recall:<8} {milvus_recall:<8} {milvus_mixed_recall:<10} {qdrant_recall:<8} {qdrant_mixed_recall:<10} {chroma_recall:<8} {chroma_mixed_recall:<12}")

    print(f"\n{'='*140}")
    print(f"=== (: /) ===")
    print(f"{'='*140}")
    # [Complete: add qdrant/chroma mixed]
    print(f"{'exp_factor':<12} {'beta':<6} | {'hdmg':<12} {'representative':<12} {'pyflat':<12} {'milvus':<12} {'milvus_mix':<12} {'qdrant':<12} {'qdrant_mix':<12} {'chroma':<12} {'chroma_mix':<12}")
    print("-" * 140)
    for r in results_summary:
        mixed_time = r['avg_mixed_time'] * 1000
        repr_time = r['avg_representative_time'] * 1000
        pyflat_str = f"{r['avg_python_flat_time'] * 1000:.2f}" if r['avg_python_flat_time'] is not None else "N/A"
        milvus_str = f"{r['avg_milvus_time'] * 1000:.2f}" if r['avg_milvus_time'] is not None else "N/A"
        milvus_mixed_str = f"{r.get('avg_milvus_mixed_time', 0.0) * 1000:.2f}" if r.get('avg_milvus_mixed_time') is not None else "N/A"
        qdrant_str = f"{r.get('avg_qdrant_time', 0.0) * 1000:.2f}" if r.get('avg_qdrant_time') is not None else "N/A"
        qdrant_mixed_str = f"{r.get('avg_qdrant_mixed_time', 0.0) * 1000:.2f}" if r.get('avg_qdrant_mixed_time') is not None else "N/A"
        chroma_str = f"{r.get('avg_chroma_time', 0.0) * 1000:.2f}" if r.get('avg_chroma_time') is not None else "N/A"
        chroma_mixed_str = f"{r.get('avg_chroma_mixed_time', 0.0) * 1000:.2f}" if r.get('avg_chroma_mixed_time') is not None else "N/A"
        print(f"{r['expansion_factor']:<12} {r['beta']:<6.2f} | "
              f"{mixed_time:<12.2f} {repr_time:<12.2f} {pyflat_str:<12} {milvus_str:<12} {milvus_mixed_str:<12} {qdrant_str:<12} {qdrant_mixed_str:<12} {chroma_str:<12} {chroma_mixed_str:<12}")
    print(f"{'='*140}")

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
            "bench": "coco",
            "expansion_factor": EXPANSION_FACTOR,
            "alpha": ALPHA,
            "betas": betas,
            "recall": recall,
            "latency_ms": latency_ms,
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary written to {args.summary_json}", flush=True)


    if any(r.get('hdmg_avg_depth') is not None for r in results_summary):
        print(f"\n{'='*120}")
        print(f"=== HDMG ( beta) ===")
        print(f"{'='*120}")
        print(f"{'beta':<6} | {'avg_hops':<10} {'avg_depth':<10} {'cand_pool':<10} {'ms_calls':<10} | {'t_entry':<10} {'t_walk':<10} {'t_rerank':<10} {'t_total':<10}")
        print("-" * 120)
        for r in results_summary:
            hops = f"{r['hdmg_avg_hops']:.2f}" if r.get('hdmg_avg_hops') is not None else "N/A"
            depth = f"{r['hdmg_avg_depth']:.2f}" if r.get('hdmg_avg_depth') is not None else "N/A"
            pool = f"{r['hdmg_avg_candidate_pool']:.1f}" if r.get('hdmg_avg_candidate_pool') is not None else "N/A"
            ms_calls = f"{r['hdmg_avg_mixed_score_calls']:.1f}" if r.get('hdmg_avg_mixed_score_calls') is not None else "N/A"
            t_entry = f"{r['hdmg_avg_t_entry_ms']:.3f}" if r.get('hdmg_avg_t_entry_ms') is not None else "N/A"
            t_walk = f"{r['hdmg_avg_t_walk_ms']:.3f}" if r.get('hdmg_avg_t_walk_ms') is not None else "N/A"
            t_rerank = f"{r['hdmg_avg_t_rerank_ms']:.3f}" if r.get('hdmg_avg_t_rerank_ms') is not None else "N/A"
            t_total = f"{r['hdmg_avg_t_entry_ms'] + r['hdmg_avg_t_walk_ms'] + r['hdmg_avg_t_rerank_ms']:.3f}" if r.get('hdmg_avg_t_entry_ms') is not None else "N/A"
            print(f"{r['beta']:<6.2f} | {hops:<10} {depth:<10} {pool:<10} {ms_calls:<10} | {t_entry:<10} {t_walk:<10} {t_rerank:<10} {t_total:<10}")
        print(f"{'='*120}")

    total_elapsed = time.time() - main_start_time
    print(f"\n=== Total time taken for this run: {total_elapsed:.2f} ({total_elapsed/60:.2f} ) ===")
