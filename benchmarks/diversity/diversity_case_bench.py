"""
Diversity Case Bench 鈥?銆屻€併€?1锛?(SciFact) 2锛?(arXiv) 3锛?(COCO) 4锛?(LoCoMo) 5锛?(arXiv)
"""

import argparse
import os, sys, json, re, time
import numpy as np
from typing import List, Dict, Tuple, Optional

VIOLAS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(VIOLAS_ROOT, ".."))
if VIOLAS_ROOT not in sys.path:
    sys.path.insert(0, VIOLAS_ROOT)
from violas.storage import (
    VectorMap,
    VectorGroup,
    VectorRef,
    VectorRelation,
    add_relation_to_description,
    get_relations_from_description,
)
from violas.storage.utils import cosine_distance, cosine_similarity

# ============= Lightweight vectorizer =============

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class MiniVectorizer:
    def __init__(self, dim=128, max_features=5000):
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.dim = dim

    def fit_transform(self, texts):
        X = self.tfidf.fit_transform(texts)
        V = self.svd.fit_transform(X).astype(np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return V / norms

    def transform(self, texts):
        X = self.tfidf.transform(texts)
        V = self.svd.transform(X).astype(np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return V / norms

def split_sentences(text, num_segs=3):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= num_segs:
        return [s for s in sents if s.strip()]
    seg_size = len(sents) // num_segs
    segs = []
    for i in range(num_segs):
        start = i * seg_size
        end = start + seg_size if i < num_segs - 1 else len(sents)
        segs.append(" ".join(sents[start:end]))
    return [s for s in segs if s.strip()]


# ============================================================
# Scenario 1: Version chain retrieval (SciFact)
# ============================================================

def scenario_1_version_chain():
    print("\n" + "=" * 70)
    print("Scenario 1: Version chain retrieval 鈥?SciFact Document version chain + time constraints")
    print("=" * 70)

    with open(os.path.join(WORKSPACE_ROOT, "dataset/scifact/corpus.jsonl")) as f:
        docs = [json.loads(line) for line in f]

    long_docs = [d for d in docs if len(d["text"]) > 800][:50]
    print(f"Select {len(long_docs)} Long document construction version chain")

    all_texts = []
    version_info = []

    effective_dates = [
        ("2020-01", "2021-06"),
        ("2021-06", "2023-03"),
        ("2023-03", "2025-12"),
    ]

    for doc in long_docs:
        segs = split_sentences(doc["text"], 3)
        if len(segs) < 3:
            continue
        doc_id = str(doc["_id"])
        for vi, seg in enumerate(segs):
            all_texts.append(seg)
            version_info.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "version": vi + 1,
                "text": seg,
                "effective_from": effective_dates[vi][0],
                "effective_to": effective_dates[vi][1],
            })

    vec = MiniVectorizer(dim=128)
    vectors = vec.fit_transform(all_texts)

    vm = VectorMap()
    vi_idx = 0
    for doc in long_docs:
        doc_id = str(doc["_id"])
        doc_versions = [v for v in version_info if v["doc_id"] == doc_id]
        if len(doc_versions) < 3:
            continue

        doc_vecs = vectors[vi_idx:vi_idx + len(doc_versions)]
        descs = []
        for i, vinfo in enumerate(doc_versions):
            desc = {
                "doc_id": doc_id,
                "version": vinfo["version"],
                "effective_from": vinfo["effective_from"],
                "effective_to": vinfo["effective_to"],
                "text": vinfo["text"][:200],
            }
            descs.append(desc)

        g = VectorGroup(
            group_name="versions",
            representative=np.mean(doc_vecs, axis=0),
            rep_description=f"Doc {doc_id} versions",
            vectors=list(doc_vecs),
            descriptions=descs,
            vector_type="text"
        )
        vm.insert(doc_id, g)
        vi_idx += len(doc_versions)

    for doc_id in vm.data:
        groups = vm.data[doc_id]["groups"]
        for g in groups:
            for i in range(len(g.descriptions)):
                if i > 0:
                    prev_ref = VectorRef(doc_id, "versions", i - 1)
                    add_relation_to_description(g.descriptions[i],
                        VectorRelation(prev_ref, "PREV_VERSION"))
                if i < len(g.descriptions) - 1:
                    next_ref = VectorRef(doc_id, "versions", i + 1)
                    add_relation_to_description(g.descriptions[i],
                        VectorRelation(next_ref, "NEXT_VERSION"))

    vm.build_index()

    query_text = "apparent diffusion coefficient measurement in cerebral white matter"
    query_as_of = "2022-01"
    query_vec = vec.transform([query_text])[0]

    results = vm.search(query_vec, mode="single", top_k=5)

    print(f"\nQuery: \"{query_text}\"")
    print(f"As-of date: {query_as_of}")
    print(f"\n--- Top-K () ---")
    for i, sr in enumerate(results):
        desc = sr.group.descriptions[sr.vector_idx]
        print(f"  #{i+1} doc={desc['doc_id']} v{desc['version']} "
              f"effective=[{desc['effective_from']}, {desc['effective_to']}) "
              f"dist={sr.distance:.4f}")
        print(f"       text: {desc['text'][:120]}...")

    print(f"\n--- ( as_of={query_as_of} ) ---")
    hit = results[0]
    desc = hit.group.descriptions[hit.vector_idx]
    print(f"  Initial hit: doc={desc['doc_id']} v{desc['version']} "
          f"effective=[{desc['effective_from']}, {desc['effective_to']})")

    current_desc = desc
    current_idx = hit.vector_idx
    found = False
    visited = set()
    while not found:
        ef = current_desc["effective_from"]
        et = current_desc["effective_to"]
        if ef <= query_as_of < et:
            print(f"  鉁?Valid version found: v{current_desc['version']} "
                  f"effective=[{ef}, {et})")
            print(f"    text: {current_desc['text'][:200]}...")
            found = True
            break

        visited.add(current_idx)
        rels = get_relations_from_description(current_desc, "PREV_VERSION")
        if not rels:
            rels = get_relations_from_description(current_desc, "NEXT_VERSION")

        moved = False
        for rel in rels:
            ref = rel.ref
            target_idx = ref.vector_idx
            if target_idx not in visited:
                resolved = vm.get_vector_by_ref(ref)
                if resolved:
                    current_desc = resolved[1]
                    current_idx = target_idx
                    print(f"  鈫?jump to v along relation{current_desc['version']} "
                          f"effective=[{current_desc['effective_from']}, {current_desc['effective_to']})")
                    moved = True
                    break
        if not moved:
            print(f"  脳 Unable to continue backtracking")
            break

    case_data = {
        "query": query_text,
        "as_of": query_as_of,
        "initial_hit": {
            "doc_id": desc["doc_id"],
            "version": desc["version"],
            "effective": [desc["effective_from"], desc["effective_to"]],
            "text": desc["text"][:300]
        },
        "final_version": {
            "version": current_desc["version"],
            "effective": [current_desc["effective_from"], current_desc["effective_to"]],
            "text": current_desc["text"][:300]
        }
    }
    return case_data


# ============================================================
# Scenario 2: Multi-hop heterogeneous relationship expansion (arXiv)
# ============================================================

def scenario_2_multi_hop_expand():
    print("\n" + "=" * 70)
    print("Scenario 2: Multi-hop heterogeneous relationships unfold 鈥?arXiv The paper expands from the conclusion paragraph to a variety of evidence")
    print("=" * 70)

    with open(os.path.join(WORKSPACE_ROOT, "dataset/arxiv/9samples_arxiv.json")) as f:
        papers = json.load(f)

    excluded = {"arxiv_id", "title", "Title", "metadata", "query", "context", "authors"}
    all_texts = []
    text_info = []
    meta_texts = []
    meta_keys = []

    for paper in papers:
        title = paper.get("title", "")
        meta = paper.get("metadata", "")
        if meta:
            meta_texts.append(meta)
            meta_keys.append(title)
        for key, val in paper.items():
            if key not in excluded and isinstance(val, str) and len(val) > 20:
                segs = split_sentences(val, 3)
                for si, seg in enumerate(segs):
                    all_texts.append(seg)
                    text_info.append({"key": title, "group": key, "seg_idx": si, "text": seg})

    vec = MiniVectorizer(dim=128)
    vectors = vec.fit_transform(all_texts + meta_texts)
    text_vectors = vectors[:len(all_texts)]
    meta_vectors = vectors[len(all_texts):]

    vm = VectorMap()
    key_vecs = {}
    for i, mk in enumerate(meta_keys):
        key_vecs[mk] = meta_vectors[i]

    for i, info in enumerate(text_info):
        key = info["key"]
        gname = info["group"]
        g = vm.get_group_by_name(key, gname)
        desc = {"key": key, "group_name": gname, "segment_idx": info["seg_idx"],
                "text": info["text"][:300]}
        if not g:
            g = VectorGroup(gname, text_vectors[i], "first",
                            [text_vectors[i]], [desc], "text")
            vm.insert(key, g)
        else:
            g.append(text_vectors[i], desc)

    vm.set_key_vectors(key_vecs)

    RELATION_MAP = {
        "Abstract": "CONCLUSION_FROM_ABSTRACT",
        "Introduction": "CLAIM_BACKGROUND",
        "CapsNet architecture": "METHOD_ARCHITECTURE",
        "Capsules on MNIST": "RESULT_EXPERIMENT",
        "Discussion and previous work": "CLAIM_DISCUSSION",
    }

    for key in vm.data:
        groups = vm.data[key]["groups"]
        target_groups = {g.group_name: g for g in groups}
        for g in groups:
            for vi in range(len(g.descriptions)):
                for tg_name, rel_type in RELATION_MAP.items():
                    if tg_name != g.group_name and tg_name in target_groups:
                        tg = target_groups[tg_name]
                        ref = VectorRef(key, tg_name, 0)
                        add_relation_to_description(g.descriptions[vi],
                            VectorRelation(ref, rel_type))

    vm.build_index()

    query = "How does routing by agreement work in capsule networks?"
    q_vec = vec.transform([query])[0]
    hits = vm.search(q_vec, mode="single", top_k=3)

    print(f"\nQuery: \"{query}\"")
    print(f"\n--- Top-3 Hit ---")
    for i, sr in enumerate(hits):
        desc = sr.group.descriptions[sr.vector_idx]
        print(f"  #{i+1} [{desc['group_name']}] dist={sr.distance:.4f}")
        print(f"       {desc['text'][:120]}...")

    hit = hits[0]
    hit_desc = hit.group.descriptions[hit.vector_idx]
    all_rels = get_relations_from_description(hit_desc)

    print(f"\n--- ({len(all_rels)} ) ---")
    expanded = []
    for rel in all_rels:
        resolved = vm.get_vector_by_ref(rel.ref)
        if resolved:
            vec_data, rdesc, _ = resolved
            print(f"  [{rel.relation_type}] 鈫?{rel.ref.group_name}[{rel.ref.vector_idx}]")
            print(f"     {rdesc.get('text', '')[:120]}...")
            expanded.append({
                "relation_type": rel.relation_type,
                "target_group": rel.ref.group_name,
                "text": rdesc.get("text", "")[:300]
            })

    case_data = {
        "query": query,
        "hit": {"group": hit_desc["group_name"], "text": hit_desc["text"][:300]},
        "expanded_nodes": expanded
    }
    return case_data


# ============================================================
# Scenario 3: Back Reference Attribution (COCO)
# ============================================================

def scenario_3_reverse_reference():
    print("\n" + "=" * 70)
    print("Scenario 3: Back reference tracing 鈥?COCO Reverse the image to find all captions that refer to it")
    print("=" * 70)

    with open(os.path.join(WORKSPACE_ROOT, "dataset/coco/coco_dataset_40.json")) as f:
        items = json.load(f)

    all_cap_texts = []
    cap_info = []
    img_ids = []

    for idx, item in enumerate(items):
        img_ids.append(idx)
        for ci, cap in enumerate(item["text"]):
            all_cap_texts.append(cap)
            cap_info.append({"img_idx": idx, "cap_idx": ci,
                             "path": item["path"], "text": cap})

    vec = MiniVectorizer(dim=128)
    cap_vectors = vec.fit_transform(all_cap_texts)

    vm = VectorMap()
    for idx, item in enumerate(items):
        img_path = item["path"]
        cat_key = f"img_{idx:03d}"

        img_desc = {"img_idx": idx, "path": img_path, "type": "image"}
        img_dummy_vec = np.mean(
            cap_vectors[[i for i, c in enumerate(cap_info) if c["img_idx"] == idx]],
            axis=0)

        img_group = VectorGroup("images", img_dummy_vec, "mean",
                                [img_dummy_vec], [img_desc], "image")
        vm.insert(cat_key, img_group)

        this_caps = [(i, c) for i, c in enumerate(cap_info) if c["img_idx"] == idx]
        cap_descs = []
        cap_vecs_list = []
        for vi, (gi, cinfo) in enumerate(this_caps):
            desc = {"img_idx": idx, "cap_idx": cinfo["cap_idx"],
                    "path": img_path, "text": cinfo["text"]}
            img_ref = VectorRef(cat_key, "images", 0)
            add_relation_to_description(desc,
                VectorRelation(img_ref, "CAPTION_IMAGE"))
            cap_descs.append(desc)
            cap_vecs_list.append(cap_vectors[gi])

        if cap_vecs_list:
            cap_group = VectorGroup("captions", np.mean(cap_vecs_list, axis=0), "mean",
                                    cap_vecs_list, cap_descs, "text")
            vm.insert(cat_key, cap_group)

    vm.build_index()

    target_key = "img_005"
    target_path = items[5]["path"]
    target_ref_str = f"{target_key}%images%0"

    print(f"\nTarget image: {target_path} (key={target_key})")
    print(f"target ref: {target_ref_str}")
    print(f"\n--- Back reference scanning: traverse all descriptions to find captions that reference this image ---")

    referrers = []
    for key, entry in vm.data.items():
        for g in entry["groups"]:
            for vi, desc in enumerate(g.descriptions):
                rels = get_relations_from_description(desc, "CAPTION_IMAGE")
                for rel in rels:
                    if rel.ref.to_string() == target_ref_str:
                        referrers.append({
                            "key": key,
                            "group": g.group_name,
                            "vector_idx": vi,
                            "text": desc.get("text", ""),
                            "relation_type": rel.relation_type
                        })

    print(f"\nturn up {len(referrers)} captions referencing this image:")
    for r in referrers:
        print(f"  [{r['relation_type']}] {r['key']}/{r['group']}[{r['vector_idx']}]")
        print(f"    \"{r['text']}\"")

    print(f"\n--- 锛?ANN search ( caption) ---")
    target_img_vec = vm.data[target_key]["groups"][0].vectors[0]
    ann_hits = vm.search(target_img_vec, group_name="captions", mode="single", top_k=5)
    print(f"  ANN Top-5 result:")
    for i, sr in enumerate(ann_hits):
        desc = sr.group.descriptions[sr.vector_idx]
        is_correct = desc.get("img_idx") == 5
        print(f"    #{i+1} img_idx={desc.get('img_idx')} dist={sr.distance:.4f} "
              f"{'鉁? if is_correct else '鉁?} \"{desc.get('text', '')[:80]}\"")

    case_data = {
        "target_image": target_path,
        "target_ref": target_ref_str,
        "reverse_referrers": referrers,
        "ann_top5": [{"img_idx": sr.group.descriptions[sr.vector_idx].get("img_idx"),
                       "text": sr.group.descriptions[sr.vector_idx].get("text", ""),
                       "dist": sr.distance} for sr in ann_hits]
    }
    return case_data


# ============================================================
# Scenario 4: Branch session retrieval (LoCoMo)
# ============================================================

def scenario_4_branch_conversation():
    print("\n" + "=" * 70)
    print("Scenario 4: Branch session retrieval 鈥?LoCoMo Backtrack by topic branch")
    print("=" * 70)

    with open(os.path.join(WORKSPACE_ROOT, "dataset/locomo/locomo10.json")) as f:
        samples = json.load(f)

    sample = samples[0]
    sample_id = sample["sample_id"]
    summaries = sample["session_summary"]
    conv = sample["conversation"]

    BRANCHES = {
        "branch_lgbtq": [1, 3, 5, 6, 7, 9, 10, 12],
        "branch_family": [4, 8, 11, 13, 15, 17, 19],
        "branch_outdoor": [2, 14, 16, 18],
    }
    BRANCH_LABELS = {
        "branch_lgbtq": "LGBTQ advocacy & identity",
        "branch_family": "Family, adoption & kids",
        "branch_outdoor": "Outdoor activities & travel",
    }

    all_texts = []
    seg_info = []

    for bname, session_nums in BRANCHES.items():
        for snum in session_nums:
            skey = f"session_{snum}_summary"
            if skey not in summaries:
                continue
            text = summaries[skey]
            segs = split_sentences(text, 2)
            for si, seg in enumerate(segs):
                all_texts.append(seg)
                seg_info.append({
                    "branch": bname, "session_num": snum,
                    "seg_idx": si, "text": seg
                })

    vec = MiniVectorizer(dim=128)
    vectors = vec.fit_transform(all_texts)

    vm = VectorMap()
    key = sample_id

    branch_data = {}
    for i, info in enumerate(seg_info):
        bname = info["branch"]
        if bname not in branch_data:
            branch_data[bname] = {"vecs": [], "descs": []}
        desc = {
            "branch": bname,
            "branch_label": BRANCH_LABELS[bname],
            "session_num": info["session_num"],
            "seg_idx": info["seg_idx"],
            "text": info["text"][:300],
        }
        branch_data[bname]["vecs"].append(vectors[i])
        branch_data[bname]["descs"].append(desc)

    for bname, data in branch_data.items():
        g = VectorGroup(bname, np.mean(data["vecs"], axis=0), "mean",
                        data["vecs"], data["descs"], "text", "branch")
        vm.insert(key, g)

    groups_by_name = {g.group_name: g for g in vm.data[key]["groups"]}

    for bname in BRANCHES:
        g = groups_by_name[bname]
        for vi in range(len(g.descriptions)):
            if vi > 0:
                ref = VectorRef(key, bname, vi - 1)
                add_relation_to_description(g.descriptions[vi],
                    VectorRelation(ref, "PREV_IN_BRANCH"))

    family_g = groups_by_name["branch_family"]
    last_vi = len(family_g.descriptions) - 1
    for bname in BRANCHES:
        g = groups_by_name[bname]
        for vi in range(len(g.descriptions)):
            ref = VectorRef(key, "branch_family", last_vi)
            add_relation_to_description(g.descriptions[vi],
                VectorRelation(ref, "RESOLVES"))

    vm.build_index()

    query_text = "What progress has Caroline made toward adoption?"
    q_vec = vec.transform([query_text])[0]

    hits = vm.search(q_vec, mode="single", top_k=5)

    print(f"\nQuery: \"{query_text}\"")
    print(f"\n--- Top-5 Hit ---")
    for i, sr in enumerate(hits):
        desc = sr.group.descriptions[sr.vector_idx]
        print(f"  #{i+1} [{desc['branch']}] session_{desc['session_num']}[seg {desc['seg_idx']}] "
              f"dist={sr.distance:.4f}")
        print(f"       {desc['text'][:120]}...")

    hit = hits[0]
    hit_desc = hit.group.descriptions[hit.vector_idx]

    print(f"\n--- Branch backtracing: only along {hit_desc['branch']} ({hit_desc['branch_label']}) Traceback ---")
    path = [(hit_desc["branch"], hit_desc["session_num"], hit_desc["seg_idx"],
             hit_desc["text"][:120])]

    current_desc = hit_desc
    for step in range(10):
        rels = get_relations_from_description(current_desc, "PREV_IN_BRANCH")
        if not rels:
            break
        ref = rels[0].ref
        resolved = vm.get_vector_by_ref(ref)
        if not resolved:
            break
        _, rdesc, _ = resolved
        path.append((rdesc["branch"], rdesc["session_num"], rdesc["seg_idx"],
                      rdesc["text"][:120]))
        current_desc = rdesc

    print(f"  ({len(path)} , {hit_desc['branch']} ):")
    for step, (br, snum, si, txt) in enumerate(path):
        prefix = "  鈽?" if step == 0 else "  鈫?"
        print(f"  {prefix}session_{snum}[seg {si}] {txt}...")

    print(f"\n--- Comparison: Linear time series backtracking (no distinction between branches) ---")
    hit_snum = hit_desc["session_num"]
    linear_path = []
    for snum in range(hit_snum, max(0, hit_snum - 6), -1):
        skey = f"session_{snum}_summary"
        if skey in summaries:
            text = summaries[skey][:120]
            which_branch = "?"
            for bname, nums in BRANCHES.items():
                if snum in nums:
                    which_branch = BRANCH_LABELS[bname]
                    break
            linear_path.append((snum, which_branch, text))

    print(f"  linear forward {len(linear_path)} sessions:")
    for snum, br, txt in linear_path:
        marker = "鉁? if "Family" in br or "adoption" in br.lower() else "鉁?irrelevant"
        print(f"    session_{snum} [{br}] {marker}")
        print(f"      {txt}...")

    case_data = {
        "query": query_text,
        "hit_branch": hit_desc["branch"],
        "hit_branch_label": hit_desc["branch_label"],
        "branch_path": [(snum, txt[:100]) for _, snum, _, txt in path],
        "linear_path": [(snum, br) for snum, br, _ in linear_path],
    }
    return case_data


# ============================================================
# Scenario 5: Combined entity retrieval (arXiv)
# ============================================================

def scenario_5_composite_entity():
    print("\n" + "=" * 70)
    print("Scenario 5: Combined entity retrieval 鈥?arXiv After the paper is hit, the complete entity (all sections) is returned")
    print("=" * 70)

    with open(os.path.join(WORKSPACE_ROOT, "dataset/arxiv/9samples_arxiv.json")) as f:
        papers = json.load(f)

    excluded = {"arxiv_id", "title", "Title", "metadata", "query", "context", "authors"}

    all_texts = []
    text_info = []
    meta_texts = []
    meta_keys = []

    for paper in papers:
        title = paper.get("title", "")
        meta = paper.get("metadata", "")
        if meta:
            meta_texts.append(meta)
            meta_keys.append(title)
        for k, v in paper.items():
            if k not in excluded and isinstance(v, str) and len(v) > 20:
                all_texts.append(v[:500])
                text_info.append({"key": title, "group": k, "text": v[:500]})

    vec = MiniVectorizer(dim=128)
    vectors = vec.fit_transform(all_texts + meta_texts)
    text_vectors = vectors[:len(all_texts)]
    meta_vectors = vectors[len(all_texts):]

    vm = VectorMap()
    key_vecs = {}
    for i, mk in enumerate(meta_keys):
        key_vecs[mk] = meta_vectors[i]

    for i, info in enumerate(text_info):
        k = info["key"]
        gname = info["group"]
        desc = {"key": k, "group_name": gname, "text": info["text"][:300]}
        g = vm.get_group_by_name(k, gname)
        if not g:
            g = VectorGroup(gname, text_vectors[i], "first",
                            [text_vectors[i]], [desc], "text")
            vm.insert(k, g)
        else:
            g.append(text_vectors[i], desc)

    vm.set_key_vectors(key_vecs)
    vm.build_index()

    query = "capsule network architecture and routing algorithm"
    q_vec = vec.transform([query])[0]

    print(f"\nQuery: \"{query}\"")
    print(f"\n--- Stage 1: representative search 鈫?Hit key ---")
    key_hits = vm.search(q_vec, mode="representative", top_k=3)
    for i, sr in enumerate(key_hits):
        print(f"  #{i+1} key=\"{sr.key}\" dist={sr.distance:.4f}")

    best_key = key_hits[0].key
    print(f"\n--- Phase 2: Expand key=\"{best_key}\" Next all group ---")

    entity_groups = vm.data[best_key]["groups"]
    entity_data = []
    for g in entity_groups:
        print(f"\n  馃搧 Group: {g.group_name} ({len(g.vectors)} vectors)")
        for vi, desc in enumerate(g.descriptions):
            txt = desc.get("text", "")[:150]
            print(f"      [{vi}] {txt}...")
            entity_data.append({
                "group": g.group_name,
                "idx": vi,
                "text": desc.get("text", "")[:300]
            })

    print(f"\n  鉁?Return the complete entity: {len(entity_groups)} group, "
          f"{sum(len(g.vectors) for g in entity_groups)} bar vector")

    case_data = {
        "query": query,
        "matched_key": best_key,
        "groups": [{
            "name": g.group_name,
            "num_vectors": len(g.vectors),
            "preview": g.descriptions[0].get("text", "")[:200] if g.descriptions else ""
        } for g in entity_groups]
    }
    return case_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Structured retrieval diversity case bench")
    parser.add_argument(
        "--output",
        default=os.path.join(VIOLAS_ROOT, "case", "diversity_cases.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Diversity Case Bench: Five structured search paradigms")
    print("=" * 70)

    results = {}

    for name, fn in [
        ("Version chain search", scenario_1_version_chain),
        ("Multi-hop heterogeneous relationship expansion", scenario_2_multi_hop_expand),
        ("Back reference tracing", scenario_3_reverse_reference),
        ("Branch session retrieval", scenario_4_branch_conversation),
        ("Combined entity search", scenario_5_composite_entity),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\nResults have been saved to: {out_path}")


if __name__ == "__main__":
    main()
