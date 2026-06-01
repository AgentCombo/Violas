"""VectorMap storage and structured retrieval implementation."""

import uuid
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans

try:
    from .vectorgroup import VectorGroup
    from .utils import *
    from .utils import VectorRef, VectorRelation, get_relations_from_description
except ImportError:
    from vectorgroup import VectorGroup
    from utils import *
    from utils import VectorRef, VectorRelation, get_relations_from_description


class SearchResult:
    def __init__(self, distance: float, key: str, group: VectorGroup, vector_idx: Optional[int]):
        self.distance = distance
        self.key = key
        self.group = group
        self.vector_idx = vector_idx


    def __lt__(self, other):
        return self.distance < other.distance

    def __str__(self):
        return f"SearchResult(distance={self.distance}, key={self.key}, group={self.group.group_name}, vector_idx={self.vector_idx})"

    def __eq__(self, other):
        if not isinstance(other, SearchResult):
            return False
        return (self.distance == other.distance and
                self.key == other.key and
                self.group == other.group and
                self.vector_idx == other.vector_idx)


class VectorMap:

    def __init__(self):
        self.data: Dict[str, Dict] = {}
        self.context_map = {}

        # FAISS index acceleration
        self._rep_index_valid = False
        self._faiss_rep_index = None
        self._rep_index_mapping = []      # [(key, group), ...]

        self._single_index_valid = False
        self._faiss_single_index = None
        self._single_index_mapping = []   # [(key, group, vector_idx), ...]

        # Key vector storage (for hybrid search)
        # key_vectors: {base_key: np.ndarray} stores the key vector corresponding to each base_key
        # For example: {"accordion": np.array([...]), "airplane": np.array([...])}
        self.key_vectors: Dict[str, np.ndarray] = {}

        # HDMG graph index (hierarchical dynamic metric graph for accelerating hybrid queries)
        self._hdmg_nodes: List[Tuple[str, VectorGroup]] = []  # node_id -> (key, group)
        self._hdmg_base_keys: List[str] = []                 # node_id -> base_key
        self._hdmg_embedding_edges: List[List[int]] = []      # node_id -> list of neighbor node_ids
        self._hdmg_semantic_edges: List[List[int]] = []       # node_id -> list of neighbor node_ids
        self._hdmg_key_entries: Dict[str, int] = {}           # base_key -> representative node_id
        self._hdmg_key_entry_ids: List[int] = []              # representative node_ids for all base_keys
        self._hdmg_key_entry_base_keys: List[str] = []        # base_key aligned with _hdmg_key_entry_ids
        self._hdmg_key_entry_key_matrix: Optional[np.ndarray] = None  # [K, D] normalized key vectors
        self._hdmg_key_entry_rep_matrix: Optional[np.ndarray] = None  # [K, D] normalized rep vectors
        self._hdmg_global_entry: Optional[int] = None         # node_id for global entry (embedding-heavy)
        self._hdmg_built: bool = False
        self._hdmg_last_search_stats: Dict[str, Any] = {}

    # ============= Insert vector group, add vector =============
    def insert(self,
               key: str,
               vector_group: VectorGroup,
               metadata: Optional[Dict] = None) -> None:
        if key not in self.data:
            self.data[key] = {
                'metadata': metadata or {},
                'groups': []
            }

        # Initialize collections of vector types and group types
        if 'vector_types' not in self.data[key]['metadata']:
            self.data[key]['metadata']['vector_types'] = set()
        self.data[key]['metadata']['vector_types'].add(vector_group.vector_type)

        if 'group_types' not in self.data[key]['metadata']:
            self.data[key]['metadata']['group_types'] = set()
        self.data[key]['metadata']['group_types'].add(vector_group.group_type)

        # Initialize relationship information
        if 'relations' not in self.data[key]['metadata']:
            self.data[key]['metadata']['relations'] = {
                'pair': [],    # Strong correlation (two vector groups, one-to-one correspondence between vectors)
                'tree': []     # Hierarchical relationships (parent-child node pairs)
            }

        # Check if the group name already exists
        for group in self.data[key]['groups']:
            if group.group_name == vector_group.group_name:
                print(f": '{vector_group.group_name}' ，", flush=True)
                self.data[key]['groups'] = [g for g in self.data[key]['groups']
                                          if g.group_name != vector_group.group_name]
                break

        self.data[key]['groups'].append(vector_group)

        self._rep_index_valid = False
        self._single_index_valid = False

    def set_key_vectors(self, key_vectors: Dict[str, np.ndarray]) -> None:
        """
        key （） Args: key_vectors: {base_key: np.ndarray} - base_key: key （ "accordion"），- np.ndarray: key （ KeyPredictor text_features）
        """
        self.key_vectors = key_vectors
        print(f"Set key vectors for {len(key_vectors)} keys", flush=True)

    def set_key_vectors_from_predictor(self, predictor) -> None:
        """
        KeyPredictor key Args: predictor: KeyPredictor ， keys text_features
        """
        key_vectors = {}
        for i, key in enumerate(predictor.keys):
            key_vectors[key] = predictor.text_features[i]
        self.key_vectors = key_vectors
        print(f"KeyPredictor {len(key_vectors)} key", flush=True)

    def get_key_vector(self, key: str) -> Optional[np.ndarray]:
        """
        key key Args: key: key （ key， "accordion-0001"，base_key） Returns: key ， None
        """
        import re
        # Extract base_key (remove only the last -number suffix, such as "comp.os.ms-windows.misc-0001" -> "comp.os.ms-windows.misc")
        match = re.match(r'^(.+)-\d+$', key)
        base_key = match.group(1) if match else key
        return self.key_vectors.get(base_key)

    def insert_with_auto_cluster(self,
               key: str,
               vector_group: VectorGroup,
               metadata: Optional[Dict] = None,
               alpha: float = 0.5) -> int:
        """
        VectorGroup Args: key: vector_group: metadata: alpha: (0-1) - alpha=0: （n_clusters = n）
                   - alpha=0.5: （n_clusters ≈ √n）
                   - alpha=1: （n_clusters = 1）
                   : n_clusters = n^(1-alpha)

        Returns:
            int: Number of created subgroups
        """
        vectors = vector_group.vectors
        descriptions = vector_group.descriptions
        n_vectors = len(vectors)

        # Constrain alpha to be in the range [0, 1]
        alpha = max(0.0, min(1.0, alpha))

        # Calculate the number of clusters: n_clusters = n^(1-alpha), at least 1, at most n
        n_clusters = max(1, min(n_vectors, round(n_vectors ** (1 - alpha))))

        # # Simplify the grouping logic: vectors greater than 20 are divided into 3 groups, otherwise 1 group
        # n_clusters = 3 if n_vectors > 20 else 1

        # If the number of vectors is too small and clustering is not needed, insert them directly.
        if n_clusters == 1 or n_vectors < 5:
            sub_key = f"{key}-0001"
            self.insert(sub_key, vector_group, metadata)
            return 1

        # Convert vector to numpy array for use with KMeans
        vectors_array = np.array([v if isinstance(v, np.ndarray) else np.array(v) for v in vectors])

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors_array)

        # Organize data into clusters
        clusters_data = {}  # {cluster_id: {'vectors': [], 'descriptions': []}}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters_data:
                clusters_data[label] = {'vectors': [], 'descriptions': []}
            clusters_data[label]['vectors'].append(vectors[idx])
            clusters_data[label]['descriptions'].append(descriptions[idx])

        # Create a VectorGroup for each cluster and insert
        created_count = 0
        for cluster_id in sorted(clusters_data.keys()):
            cluster_data = clusters_data[cluster_id]
            cluster_vectors = cluster_data['vectors']
            cluster_descriptions = cluster_data['descriptions']

            if len(cluster_vectors) == 0:
                continue

            # Calculate the representative vector (mean) of the cluster
            rep_vec = np.mean(np.array(cluster_vectors), axis=0)

            # Subgroup naming: key-0001, key-0002, ...
            sub_key = f"{key}-{created_count + 1:04d}"

            # Create subgroup
            sub_group = VectorGroup(
                group_name=f"{vector_group.group_name}_cluster_{created_count + 1:04d}",
                representative=rep_vec,
                rep_description=f"cluster_mean_{created_count + 1}",
                vectors=cluster_vectors,
                descriptions=cluster_descriptions,
                vector_type=vector_group.vector_type,
                group_type=vector_group.group_type
            )

            # Insert subgroup
            self.insert(sub_key, sub_group, metadata)
            created_count += 1

        return created_count

    def add_vector(self, vector: np.ndarray, description: Dict[str, Any]) -> int:
        group = self.get_group_by_name(description['key'], description['group_name'])
        if not group:
            print(f"warning: vector_map key '{description['key']}' group_name '{description['group_name']}'", flush=True)
            return -1

        return group.append(vector, normalize_description(description))

    def add_vector_list(self, vector_list: List[np.ndarray], description_list: List[Dict[str, Any]]) -> int:
        global_id = uuid.uuid4()
        idx = 0

        for vector, description in zip(vector_list, description_list):
            # normalized description
            normalized_desc = normalize_description(description)
            cur_id = f'{global_id}_{idx}'
            insert_idx = self.add_vector(vector, {**normalized_desc, 'id': cur_id})
            if insert_idx >= 0:
                idx += 1
                self.context_map[cur_id] = f"{normalized_desc['key']}%{normalized_desc['group_name']}%{insert_idx}"

        return idx

    # ============= Relationship modeling methods =============

    def add_pair_relation(self, key: str, group1_id: str, group2_id: str,
                         relation_type: str = "paired") -> bool:
        """
        （，） Args: key: group1_id: ID group2_id: ID relation_type: Returns: bool:
        """
        if key not in self.data:
            return False

        # Verify that two vector groups exist
        group1: Optional[VectorGroup] = next((g for g in self.data[key]['groups'] if g.group_id == group1_id), None)
        group2: Optional[VectorGroup] = next((g for g in self.data[key]['groups'] if g.group_id == group2_id), None)

        if not group1 or not group2:
            print(f"warning: vector group not found: {group1_id} or {group2_id}", flush=True)
            return False

        if len(group1.vectors) != len(group2.vectors):
            print(f": ，{group1.group_name}({len(group1.vectors)}) vs {group2.group_name}({len(group2.vectors)})", flush=True)
            return False

        relation = {
            'group1_id': group1_id,
            'group2_id': group2_id,
            'group1_name': group1.group_name,
            'group2_name': group2.group_name,
            'relation_type': relation_type,
            'vector_count': len(group1.vectors),
        }

        self.data[key]['metadata']['relations']['pair'].append(relation)
        return True

    def add_tree_relation(self, key: str, parent_name: str, child_name: str,
                         parent_id: Optional[str] = None, child_id: Optional[str] = None,
                         relation_data: Optional[Dict] = None) -> bool:
        """
        （） Args: parent_name: child_name: parent_id: ID（） child_id: ID（） relation_data: Returns: bool:
        """
        if key not in self.data:
            return False

        relation = {
            'parent_name': parent_name,
            'child_name': child_name,
            'parent_id': parent_id,
            'child_id': child_id,
        }

        if relation_data:
            relation.update(relation_data)

        self.data[key]['metadata']['relations']['tree'].append(relation)
        return True

    # ============= Query method =============

    def get(self, key: str, return_type: str = "all") -> Any:
        """
        Args: key: return_type: ("all", "metadata", "groups")

        Returns:
            return_type
        """
        if key not in self.data:
            return None

        if return_type == "metadata":
            return self.data[key]['metadata']
        elif return_type == "groups":
            return self.data[key]['groups']
        else:
            return self.data[key]

    def get_all_keys(self) -> List[str]:
        """Return all stored keys."""
        return list(self.data.keys())

    def get_group_by_name(self, key: str, group_name: str) -> Optional['VectorGroup']:
        """Return the requested VectorMap data."""
        if key in self.data:
            for group in self.data[key]['groups']:
                if group.group_name == group_name:
                    return group
        return None

    def get_group_by_id(self, key: str, group_id: str) -> Optional['VectorGroup']:
        """ID"""
        if key in self.data:
            for group in self.data[key]['groups']:
                if hasattr(group, 'group_id') and group.group_id == group_id:
                    return group
        return None

    # ============= Paper-aligned public API wrappers =============

    def create_group(
        self,
        key: str,
        vectors: List[np.ndarray],
        descriptions: List[Dict[str, Any]],
        group_name: Optional[str] = None,
        representative: Optional[np.ndarray] = None,
        rep_description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        vector_type: str = "general",
        group_type: str = "default",
    ) -> VectorGroup:
        """Create and register one semantic group."""
        if len(vectors) == 0:
            raise ValueError("vectors must not be empty")
        if representative is None:
            representative = np.mean(np.array(vectors), axis=0)
        group = VectorGroup(
            group_name=group_name or key,
            representative=representative,
            rep_description=rep_description or f"{key} representative",
            vectors=vectors,
            descriptions=descriptions,
            vector_type=vector_type,
            group_type=group_type,
        )
        self.insert(key, group, metadata=metadata)
        return group

    def create_cluster(
        self,
        key: str,
        vector_group: VectorGroup,
        alpha: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Create micro-clusters under a semantic key."""
        return self.insert_with_auto_cluster(key, vector_group, metadata=metadata, alpha=alpha)

    def insert_object(
        self,
        key: str,
        vector: np.ndarray,
        description: Optional[Dict[str, Any]] = None,
        group_name: Optional[str] = None,
        representative: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        vector_type: str = "general",
        group_type: str = "default",
    ) -> VectorRef:
        """Insert one object and return its address inside the VectorMap."""
        target_group_name = group_name or key
        group = self.get_group_by_name(key, target_group_name)
        desc = normalize_description(description)
        desc.setdefault("key", key)
        desc.setdefault("group_name", target_group_name)

        if group is None:
            rep = representative if representative is not None else vector
            group = self.create_group(
                key=key,
                vectors=[vector],
                descriptions=[desc],
                group_name=target_group_name,
                representative=rep,
                metadata=metadata,
                vector_type=vector_type,
                group_type=group_type,
            )
            vector_idx = 0
        else:
            vector_idx = group.append(vector, desc)

        self._rep_index_valid = False
        self._single_index_valid = False
        self._hdmg_built = False
        return VectorRef(key, group.group_name, vector_idx)

    def update_object(
        self,
        ref: Union[VectorRef, str],
        vector: Optional[np.ndarray] = None,
        description: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update one stored object by reference."""
        if isinstance(ref, str):
            ref = VectorRef.from_string(ref)
        group = self.get_group_by_name(ref.key, ref.group_name)
        if group is None or ref.vector_idx < 0 or ref.vector_idx >= len(group.vectors):
            return False
        if vector is not None:
            if vector.shape != group.vectors[ref.vector_idx].shape:
                return False
            group.vectors[ref.vector_idx] = vector
        if description is not None:
            merged = dict(group.descriptions[ref.vector_idx])
            merged.update(normalize_description(description))
            group.descriptions[ref.vector_idx] = merged
        self._rep_index_valid = False
        self._single_index_valid = False
        self._hdmg_built = False
        return True

    def update(
        self,
        ref: Union[VectorRef, str],
        vector: Optional[np.ndarray] = None,
        description: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Paper-style alias for updating one stored object."""
        return self.update_object(ref, vector=vector, description=description)

    def assign_object(
        self,
        ref: Union[VectorRef, str],
        key: str,
        group_name: Optional[str] = None,
    ) -> Optional[VectorRef]:
        """Move an object to another semantic group and return the new ref."""
        obj = self.get_vector_by_ref(ref)
        if obj is None:
            return None
        vector, description, _ = obj
        new_ref = self.insert_object(
            key=key,
            vector=vector,
            description={**description, "key": key, "group_name": group_name or key},
            group_name=group_name or key,
        )
        self.delete_object(ref)
        return new_ref

    def assign(
        self,
        ref: Union[VectorRef, str],
        key: str,
        group_name: Optional[str] = None,
    ) -> Optional[VectorRef]:
        """Paper-style alias for moving one object to another entity."""
        return self.assign_object(ref, key=key, group_name=group_name)

    def delete_object(self, ref: Union[VectorRef, str]) -> bool:
        """Delete one object by reference."""
        if isinstance(ref, str):
            ref = VectorRef.from_string(ref)
        group = self.get_group_by_name(ref.key, ref.group_name)
        if group is None or ref.vector_idx < 0 or ref.vector_idx >= len(group.vectors):
            return False
        del group.vectors[ref.vector_idx]
        del group.descriptions[ref.vector_idx]
        self._rep_index_valid = False
        self._single_index_valid = False
        self._hdmg_built = False
        return True

    def delete(self, ref: Union[VectorRef, str]) -> bool:
        """Paper-style alias for deleting one stored object."""
        return self.delete_object(ref)

    def add_relation(
        self,
        source: Union[VectorRef, str],
        target: Union[VectorRef, str],
        relation_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attach an object-level relation used by dependency expansion."""
        if isinstance(source, str):
            source = VectorRef.from_string(source)
        if isinstance(target, str):
            target = VectorRef.from_string(target)
        group = self.get_group_by_name(source.key, source.group_name)
        if group is None or source.vector_idx < 0 or source.vector_idx >= len(group.descriptions):
            return False
        relation = VectorRelation(target, relation_type, weight=weight, metadata=metadata)
        add_relation_to_description(group.descriptions[source.vector_idx], relation)
        return True

    def remove_relation(
        self,
        source: Union[VectorRef, str],
        target: Optional[Union[VectorRef, str]] = None,
        relation_type: Optional[str] = None,
    ) -> bool:
        """Remove relations from one object, optionally filtered by target/type."""
        if isinstance(source, str):
            source = VectorRef.from_string(source)
        if isinstance(target, str):
            target = VectorRef.from_string(target)
        group = self.get_group_by_name(source.key, source.group_name)
        if group is None or source.vector_idx < 0 or source.vector_idx >= len(group.descriptions):
            return False
        desc = group.descriptions[source.vector_idx]
        kept = []
        changed = False
        for rel_data in desc.get("related_vectors", []):
            rel = VectorRelation.from_dict(rel_data)
            same_target = target is None or rel.ref == target
            same_type = relation_type is None or rel.relation_type == relation_type
            if same_target and same_type:
                changed = True
            else:
                kept.append(rel_data)
        desc["related_vectors"] = kept
        return changed

    def search_entity(self, query_vector: np.ndarray, key: Optional[Union[str, List[str]]] = None,
                      top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Semantic-consistent retrieval, optionally scoped to known entities."""
        return self.search(query_vector, key=key, top_k=top_k, mode="single", **kwargs)

    def search_diverse(self, query_vector: np.ndarray, top_k: int = 5,
                       query_key_vector: Optional[np.ndarray] = None,
                       beta: float = 0.5, **kwargs) -> List[SearchResult]:
        """Diversity-oriented retrieval over representatives or HDMG."""
        if self._hdmg_built:
            return self.search_hdmg(query_vector, query_key_vector=query_key_vector,
                                    alpha=beta, top_k=top_k, **kwargs)
        return self.search_with_representative_rerank(
            query_vector, query_key_vector=query_key_vector, beta=beta, top_k=top_k, **kwargs)

    def search_dependency(self, query_vector: np.ndarray, top_k: int = 5,
                          relation_types: Optional[List[str]] = None,
                          hops: int = 1, **kwargs) -> List[SearchResult]:
        """Dependency-expanded retrieval through stored relations/context."""
        if relation_types:
            return self.search_with_relations(
                query_vector, top_k=top_k, relation_types=relation_types, **kwargs)
        return self.search_with_contextual_vectors(query_vector, top_k=top_k, num=hops, **kwargs)

    def search_modal(self, query_vectors: Dict[str, np.ndarray], top_k: int = 5,
                     modality_weights: Optional[Dict[str, float]] = None,
                     **kwargs) -> List[SearchResult]:
        """Cross-modal retrieval over related modalities."""
        return self.search_multimodal(
            query_vectors=query_vectors,
            modality_weights=modality_weights,
            top_k=top_k,
            **kwargs,
        )

    def get_relations(self, key: str) -> Dict:
        """Return all stored keys."""
        if key in self.data:
            return self.data[key]['metadata'].get('relations', {})
        return {}

    def get_hierarchy_tree(self, key: str) -> Dict:
        """
        key Returns: Dict: root_nodes, parent_children, node_groups - root_nodes: （） - parent_children: {parent: [children]}
                - node_groups: ID {node_name: [group_ids]}（）
        """
        if key not in self.data:
            return {}

        relations = self.get_relations(key)

        # Build a hierarchy
        hierarchy = {
            'root_nodes': [],
            'parent_children': {},
            'node_groups': {}  # Mapping of node names to vector group IDs
        }

        # Collect tree relationships
        for tree_rel in relations.get('tree', []):
            parent_name = tree_rel['parent_name']
            child_name = tree_rel['child_name']
            parent_id = tree_rel.get('parent_id')
            child_id = tree_rel.get('child_id')

            # Record father-son relationship
            if parent_name not in hierarchy['parent_children']:
                hierarchy['parent_children'][parent_name] = []
            hierarchy['parent_children'][parent_name].append(child_name)

            # Record the mapping of nodes to vector groups (including parent nodes and child nodes)
            if parent_id:
                if parent_name not in hierarchy['node_groups']:
                    hierarchy['node_groups'][parent_name] = []
                if parent_id not in hierarchy['node_groups'][parent_name]:
                    hierarchy['node_groups'][parent_name].append(parent_id)

            if child_id:
                if child_name not in hierarchy['node_groups']:
                    hierarchy['node_groups'][child_name] = []
                if child_id not in hierarchy['node_groups'][child_name]:
                    hierarchy['node_groups'][child_name].append(child_id)

        # Find the root node (the node that has no parent)
        all_children = set()
        for children in hierarchy['parent_children'].values():
            all_children.update(children)

        hierarchy['root_nodes'] = [node for node in hierarchy['parent_children'].keys()
                                  if node not in all_children]

        return hierarchy

    # ============= FAISS Accelerated Index =============

    def build_index(self):
        """FAISS （representative + single）"""
        if _HAS_FAISS:
            self.build_rep_index()
            self.build_single_index()

    def build_rep_index(self, **_kw):
        """FAISS Flat"""
        rep_vectors = []
        self._rep_index_mapping = []
        for key in self.data:
            for group in self.data[key]['groups']:
                if group.representative is not None:
                    rep_vectors.append(np.array(group.representative, dtype=np.float32))
                    self._rep_index_mapping.append((key, group))
        if not _HAS_FAISS or len(rep_vectors) == 0:
            self._faiss_rep_index = None
            self._rep_index_valid = True
            return
        mat = np.vstack(rep_vectors).astype(np.float32)
        mat = np.ascontiguousarray(mat)
        faiss.normalize_L2(mat)
        self._faiss_rep_index = faiss.IndexFlatIP(mat.shape[1])
        self._faiss_rep_index.add(mat)
        self._rep_index_valid = True

    def build_single_index(self):
        """FAISS Flat （ single ）"""
        all_vectors = []
        self._single_index_mapping = []
        for key in self.data:
            for group in self.data[key]['groups']:
                for idx, vec in enumerate(group.vectors):
                    all_vectors.append(np.array(vec, dtype=np.float32))
                    self._single_index_mapping.append((key, group, idx))
        if not _HAS_FAISS or len(all_vectors) == 0:
            self._faiss_single_index = None
            self._single_index_valid = True
            return
        mat = np.vstack(all_vectors).astype(np.float32)
        mat = np.ascontiguousarray(mat)
        faiss.normalize_L2(mat)
        self._faiss_single_index = faiss.IndexFlatIP(mat.shape[1])
        self._faiss_single_index.add(mat)
        self._single_index_valid = True

    def _faiss_search_core(self, index, mapping, query_vector, top_k,
                           group_name=None, key=None,
                           vector_type=None, group_type=None, mode="single"):
        """FAISS ： List[SearchResult] None"""
        if index is None or len(mapping) == 0:
            return None
        has_filter = any(x is not None for x in (group_name, key, vector_type, group_type))
        if has_filter:
            search_k = min(len(mapping), max(top_k * 50, 500))
        else:
            search_k = min(len(mapping), top_k)
        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        q = np.ascontiguousarray(q)
        faiss.normalize_L2(q)
        sims, idxs = index.search(q, search_k)

        # Expand key prefix
        if isinstance(key, str):
            input_keys = [key]
        elif isinstance(key, (list, tuple)):
            input_keys = list(key)
        else:
            input_keys = None
        if input_keys is not None:
            all_keys = set(self.data.keys())
            search_keys = set()
            for k in input_keys:
                if k in all_keys:
                    search_keys.add(k)
                else:
                    for dk in all_keys:
                        if dk.startswith(f"{k}-"):
                            search_keys.add(dk)
        else:
            search_keys = None

        results = []
        for sim, fi in zip(sims[0], idxs[0]):
            if fi < 0:
                continue
            if mode == "single":
                mk, mg, vi = mapping[fi]
            else:
                mk, mg = mapping[fi]
                vi = None
            if search_keys is not None and mk not in search_keys:
                continue
            if group_name and mg.group_name != group_name:
                continue
            if vector_type and mg.vector_type != vector_type:
                continue
            if group_type and mg.group_type != group_type:
                continue
            results.append(SearchResult(1.0 - float(sim), mk, mg, vi))
            if len(results) >= top_k:
                break
        if has_filter and len(results) < top_k:
            return None
        return results

    # ============= Search method =============

    def search(self,
               query_vector: Union[np.ndarray, List[np.ndarray]],
               group_name: Optional[str] = None,
               top_k: int = 5,
               mode: str = "single",
               distance_method: str = "cosine",
               group_n_distance_method: str = "hausdorff",
               key: Optional[Union[str, List[str]]] = None,
               vector_type: Optional[str] = None,
               group_type: Optional[str] = None) -> List[SearchResult]:
        """
        Args:
            query_vector: group_name: （） top_k: k mode: ("single", "representative", "group1", "groupN")
            distance_method: （ single, representative, group1 ） group_n_distance_method: groupN - "hausdorff": （） - "maxSim": maxSim （，） key: key（，） vector_type: （） group_type: （） Returns: List[SearchResult]:
        """
        if _HAS_FAISS and distance_method == "cosine" and mode in ("single", "representative"):
            idx_obj = (self._faiss_single_index if mode == "single"
                       else self._faiss_rep_index)
            mp = (self._single_index_mapping if mode == "single"
                  else self._rep_index_mapping)
            fr = self._faiss_search_core(
                idx_obj, mp, query_vector, top_k,
                group_name, key, vector_type, group_type, mode)
            if fr is not None:
                return fr

        results = []

        # Select the keys to search (supports single key or key list, supports prefix matching)
        all_keys = list(self.data.keys())

        if isinstance(key, str):
            input_keys = [key]
        elif isinstance(key, (list, tuple)):
            input_keys = list(key)
        else:
            input_keys = None  # Search all

        # Extended keys: exact match or prefix match
        if input_keys is not None:
            search_keys = []
            for k in input_keys:
                if k in self.data:
                    # exact match
                    search_keys.append(k)
                else:
                    # Prefix matching (supports accordion matching accordion-0001, accordion-0002, etc.)
                    prefix_matches = [dk for dk in all_keys if dk.startswith(f"{k}-")]
                    search_keys.extend(prefix_matches)
        else:
            search_keys = all_keys

        # Choose a distance function
        distance_func = get_distance_function(distance_method)

        for key in search_keys:
            if key not in self.data:
                continue

            for group in self.data[key]['groups']:
                # Apply filters
                if group_name and group.group_name != group_name:
                    continue
                if vector_type and group.vector_type != vector_type:
                    continue
                if group_type and group.group_type != group_type:
                    continue

                if mode == "single":
                    # Single vector mode: search all vectors
                    for idx, vec in enumerate(group.vectors):
                        dist = distance_func(query_vector, vec)
                        results.append(SearchResult(dist, key, group, idx))

                elif mode == "representative":
                    # Represents vector pattern
                    dist = distance_func(query_vector, group.representative)
                    results.append(SearchResult(dist, key, group, None))

                elif mode == "group1":
                    # Single vector query: using average similarity
                    total_dist = 0.0
                    for vec in group.vectors:
                        total_dist += distance_func(query_vector, vec)
                    avg_dist = total_dist / len(group.vectors) if group.vectors else 0.0
                    results.append(SearchResult(avg_dist, key, group, None))

                elif mode == "groupN":
                    # Group-level search mode N: multi-vector query
                    assert isinstance(query_vector, (list, tuple)), "query_vector of groupN must be a list or tuple"

                    if group_n_distance_method == "hausdorff":
                        # Use Hausdorff distance
                        dist = hausdorff_distance(query_vector, group.vectors)
                        results.append(SearchResult(dist, key, group, None))
                    else:
                        # Use maxSim strategy (sum based on minimum distance)
                        min_dist = 0.0
                        for qv in query_vector:
                            group_min = min(distance_func(qv, gv) for gv in group.vectors)
                            min_dist += group_min
                        results.append(SearchResult(min_dist, key, group, None))

        results.sort()
        return results[:top_k]

    def search_with_rep_vec(self,
               query_vector: Union[np.ndarray, List[np.ndarray]],
               top_k: int = 5,
               gruop_expansion_factor: int = 1,
               distance_method: str = "cosine",
               print_details: bool = False,
               group_name: Optional[str] = None,
               key: Optional[Union[str, List[str]]] = None,
               vector_type: Optional[str] = None,
               group_type: Optional[str] = None) -> List[SearchResult]:
        """
        ： top_k group ： k group ， group k （ k*k ） ： k*k top_k Args: query_vector: top_k: k gruop_expansion_factor: ， distance_method: print_details: （False） group_name: （） key: key（，） vector_type: （） group_type: （） Returns: List[SearchResult]:
        """
        # Step 1. Use the representative vector to find the top_k most similar groups
        groups = self.search(
            query_vector,
            top_k=top_k * gruop_expansion_factor,
            mode="representative",
            distance_method=distance_method,
            group_name=group_name,
            key=key,
            vector_type=vector_type,
            group_type=group_type
        )

        # Print the first stage: groups information
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"： {len(groups)} group", flush=True)
            print(f"{'='*60}", flush=True)
            for i, group_result in enumerate(groups):
                group = group_result.group
                print(f"  [{i+1}] key={group_result.key}, group_name={group.group_name}, "
                      f"distance={group_result.distance:.6f}, "
                      f"vector_count={group.size()}, "
                      f"vector_type={group.vector_type}, group_type={group.group_type}", flush=True)

        # Step 2. From the selected k groups, find the k most similar vectors for each group
        distance_func = get_distance_function(distance_method)
        all_vector_results = []

        for group_result in groups:
            group = group_result.group
            group_key = group_result.key

            # Calculate the distance between the query vector and all vectors in this group
            vector_distances = []
            for idx, vec in enumerate(group.vectors):
                dist = distance_func(query_vector, vec)
                vector_distances.append((dist, idx))

            # Sort by distance, take the top_k vectors
            vector_distances.sort(key=lambda x: x[0])
            top_vectors = vector_distances[:top_k]

            # Create SearchResult
            for dist, idx in top_vectors:
                all_vector_results.append(SearchResult(dist, group_key, group, idx))

        # Second stage of printing: k*k entry information
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"： {len(groups)} group {len(all_vector_results)} (k*k={len(groups)}*{top_k})", flush=True)
            print(f"{'='*60}", flush=True)
            for i, result in enumerate(all_vector_results):
                group = result.group
                desc = group.descriptions[result.vector_idx] if result.vector_idx is not None and result.vector_idx < len(group.descriptions) else {}
                print(f"  [{i+1}] key={result.key}, group_name={group.group_name}, "
                      f"vector_idx={result.vector_idx}, distance={result.distance:.6f}, "
                      f"description={desc}", flush=True)

        # Step 3. Find the top_k vectors with the smallest distance from k*k vectors
        all_vector_results.sort()
        final_results = all_vector_results[:top_k]

        # Print the third stage: the final selected k results
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"Selected top_{top_k} results from {len(all_vector_results)} candidates", flush=True)
            print(f"{'='*60}", flush=True)
            for i, result in enumerate(final_results):
                group = result.group
                desc = group.descriptions[result.vector_idx] if result.vector_idx is not None and result.vector_idx < len(group.descriptions) else {}
                print(f"  [{i+1}] key={result.key}, group_name={group.group_name}, "
                      f"vector_idx={result.vector_idx}, distance={result.distance:.6f}, "
                      f"description={desc}", flush=True)
            print(f"{'='*60}\n", flush=True)

        return final_results

    def search_with_representative_rerank(self, query_vector: np.ndarray, query_key_vector: Optional[np.ndarray] = None,
                                         beta: float = 0.0, top_k: int = 5, num_groups: Optional[int] = None,
                                         distance_method: str = "cosine") -> List[SearchResult]:
        """
        top num_groups ，， rerank top_k。 ： 1. mixed_rep = beta*sem_d(key) + (1-beta)*rep_d，mixed_rep num_groups 2. 3. mixed_score = beta*sem_d + (1-beta)*emb_d
        4. mixed_score ， top_k
        """
        if num_groups is None:
            num_groups = top_k
        distance_func = get_distance_function(distance_method)
        q_key = None
        if query_key_vector is not None and beta != 0.0:
            q_key = np.asarray(query_key_vector, dtype=np.float64).flatten()
            qk_norm = np.linalg.norm(q_key)
            if np.isclose(qk_norm, 0.0):
                qk_norm = 1.0
            q_key = q_key / qk_norm

        # The first stage: select top num_groups groups according to the mixed score of the group mixed_rep (no longer just use rep distance)
        group_scores = []
        for key in self.data:
            for group in self.data[key]['groups']:
                if group.representative is None:
                    continue
                rep_d = distance_func(query_vector, group.representative)
                key_vec = self.get_key_vector(key) if (q_key is not None and beta != 0) else None
                sem_d = 0.0
                if key_vec is not None:
                    kv = np.asarray(key_vec, dtype=np.float64).flatten()
                    kv_norm = np.linalg.norm(kv)
                    if np.isclose(kv_norm, 0.0):
                        kv_norm = 1.0
                    kv = kv / kv_norm
                    sem_d = 1.0 - float(np.dot(q_key, kv))
                mixed_rep = beta * sem_d + (1.0 - beta) * rep_d
                group_scores.append((mixed_rep, key, group))
        group_scores.sort(key=lambda x: x[0])
        selected = group_scores[:num_groups]
        if not selected:
            return []

        # The second stage: rerank all vectors in the group using mixed scores
        candidates = []
        for _, key, group in selected:
            key_vec = self.get_key_vector(key) if (q_key is not None and beta != 0) else None
            sem_d = 0.0
            if key_vec is not None:
                kv = np.asarray(key_vec, dtype=np.float64).flatten()
                kv_norm = np.linalg.norm(kv)
                if np.isclose(kv_norm, 0.0):
                    kv_norm = 1.0
                kv = kv / kv_norm
                sem_d = 1.0 - float(np.dot(q_key, kv))
            for idx, vec in enumerate(group.vectors):
                emb_d = distance_func(query_vector, vec)
                mixed = beta * sem_d + (1.0 - beta) * emb_d
                candidates.append((mixed, SearchResult(mixed, key, group, idx)))

        candidates.sort(key=lambda x: x[0])
        return [sr for _, sr in candidates[:top_k]]

    def search_with_mixed_key_rep_vec(self,
                query_vector: np.ndarray,
                query_key_vector: np.ndarray = None,
                beta: float = 0.0,  # 0.0: completely use the representative vector, 1.0: completely use the key vector
                top_k: int = 5,
                gruop_expansion_factor: int = 1,
                distance_method: str = "cosine",
                print_details: bool = False,
                ) -> List[SearchResult]:
        """
        （key + ） ： mixed_score = beta * key_score + (1-beta) * rep_score
                  top_k * expansion_factor ： top_k Args: query_vector: （512） query_key_vector: key （ KeyPredictor.text_features） None， search_with_rep_vec beta: - beta=0.0: （ search_with_rep_vec） - beta=0.5: - beta=1.0: key top_k: k gruop_expansion_factor: ， top_k * expansion_factor distance_method: print_details: Returns: List[SearchResult]:
        """
        # If there is no key vector or beta=0, it degrades to ordinary rep_vec search
        if query_key_vector is None or beta == 0.0:
            return self.search_with_rep_vec(
                query_vector,
                top_k=top_k,
                gruop_expansion_factor=gruop_expansion_factor,
                distance_method=distance_method,
                print_details=print_details,
            )

        # Check if key_vectors is set
        if len(self.key_vectors) == 0:
            print(": key_vectors， rep_vec", flush=True)
            return self.search_with_rep_vec(
                query_vector,
                top_k=top_k,
                gruop_expansion_factor=gruop_expansion_factor,
                distance_method=distance_method,
                print_details=print_details,
            )

        distance_func = get_distance_function(distance_method)
        candidate_k = top_k * gruop_expansion_factor

        # Step 1: Use FAISS to quickly obtain rep candidates (if available)
        if (_HAS_FAISS and distance_method == "cosine"
                and self._faiss_rep_index is not None
                and len(self._rep_index_mapping) > 0):
            pre_k = min(len(self._rep_index_mapping), candidate_k * 5)
            q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            q = np.ascontiguousarray(q)
            faiss.normalize_L2(q)
            sims, idxs = self._faiss_rep_index.search(q, pre_k)
            mixed_results = []
            for sim, fi in zip(sims[0], idxs[0]):
                if fi < 0:
                    continue
                mk, mg = self._rep_index_mapping[fi]
                rep_distance = 1.0 - float(sim)
                group_key_vector = self.get_key_vector(mk)
                if group_key_vector is not None:
                    key_distance = distance_func(query_key_vector, group_key_vector)
                    mixed_distance = beta * key_distance + (1 - beta) * rep_distance
                else:
                    mixed_distance = rep_distance
                mixed_results.append(SearchResult(mixed_distance, mk, mg, None))
        else:
            mixed_results = []
            for key in self.data:
                for group in self.data[key]['groups']:
                    if group.representative is None:
                        continue
                    rep_distance = distance_func(query_vector, group.representative)
                    group_key_vector = self.get_key_vector(key)
                    if group_key_vector is not None:
                        key_distance = distance_func(query_key_vector, group_key_vector)
                        mixed_distance = beta * key_distance + (1 - beta) * rep_distance
                    else:
                        mixed_distance = rep_distance
                    mixed_results.append(SearchResult(mixed_distance, key, group, None))

        mixed_results.sort()
        selected_groups = mixed_results[:candidate_k]

        # Print the first stage: groups information
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"（ beta={beta}）：turn up {len(selected_groups)} group", flush=True)
            print(f"{'='*60}", flush=True)
            for i, group_result in enumerate(selected_groups[:20]):  # Only print the first 20
                group = group_result.group
                print(f"  [{i+1}] key={group_result.key}, group_name={group.group_name}, "
                      f"mixed_distance={group_result.distance:.6f}, "
                      f"vector_count={group.size()}", flush=True)
            if len(selected_groups) > 20:
                print(f"  ... ({len(selected_groups) - 20} more groups)", flush=True)

        # Step 2: Traverse all vectors in the selected group and find top_k final answers
        all_vector_results = []

        for group_result in selected_groups:
            group = group_result.group
            group_key = group_result.key

            # Get the key vector corresponding to the group, and calculate key_distance only once in the outer loop
            group_key_vector = self.get_key_vector(group_key)
            use_key_distance = group_key_vector is not None and query_key_vector is not None
            key_distance = distance_func(query_key_vector, group_key_vector) if use_key_distance else 0.0

            # Calculate the mixture distance of the query vector to all vectors in this group
            for idx, vec in enumerate(group.vectors):
                # vector distance
                vec_distance = distance_func(query_vector, vec)

                # If there is a key vector, calculate the mixed distance; otherwise only use the vector distance
                if use_key_distance:
                    mixed_dist = beta * key_distance + (1 - beta) * vec_distance
                else:
                    mixed_dist = vec_distance

                all_vector_results.append(SearchResult(mixed_dist, group_key, group, idx))

        # Print second stage information
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"： {len(selected_groups)} group {len(all_vector_results)} vectors", flush=True)
            print(f"{'='*60}", flush=True)

        # Step 3: Find the top_k ones with the smallest distance from all vectors
        all_vector_results.sort()
        final_results = all_vector_results[:top_k]

        # Printing Stage 3: Final Result
        if print_details:
            print(f"\n{'='*60}", flush=True)
            print(f"Final top_{top_k} results", flush=True)
            print(f"{'='*60}", flush=True)
            for i, result in enumerate(final_results):
                group = result.group
                desc = group.descriptions[result.vector_idx] if result.vector_idx is not None and result.vector_idx < len(group.descriptions) else {}
                print(f"  [{i+1}] key={result.key}, group_name={group.group_name}, "
                      f"vector_idx={result.vector_idx}, distance={result.distance:.6f}, "
                      f"img_path={desc.get('img_path', 'N/A')}", flush=True)
            print(f"{'='*60}\n", flush=True)

        return final_results

    # ============= HDMG Hierarchical Dynamic Metric Graph (Accelerate Hybrid Queries) =============
    def _hdmg_base_key(self, key: str) -> str:
        """base_key， - ， HDMG"""
        import re
        match = re.match(r'^(.+)-\d+$', key)
        return match.group(1) if match else key

    def build_hdmg(self, embedding_k: int = 16,
                  semantic_intra_k: int = 4,
                  semantic_bridge_keys: int = 2,
                  semantic_bridge_per_key: int = 1,
                  use_mutual_embedding: bool = True,
                  embedding_max_distance: Optional[float] = None,
                  semantic_same_key_limit: Optional[int] = None) -> None:
        """
        HDMG ： = (key, group)，（Embedding + Semantic）。 : embedding_k: embedding （ key） semantic_intra_k: key （） semantic_bridge_keys: key（） semantic_bridge_per_key: key use_mutual_embedding: mutual-kNN embedding embedding_max_distance: embedding （None ） semantic_same_key_limit: ； semantic_intra_k
        """
        if semantic_same_key_limit is not None:
            semantic_intra_k = semantic_same_key_limit

        self._hdmg_nodes = []
        self._hdmg_base_keys = []
        self._hdmg_key_entries = {}
        self._hdmg_key_entry_ids = []
        self._hdmg_key_entry_base_keys = []
        self._hdmg_key_entry_key_matrix = None
        self._hdmg_key_entry_rep_matrix = None
        for key in self.data:
            for group in self.data[key]['groups']:
                if group.representative is None:
                    continue
                self._hdmg_nodes.append((key, group))
                self._hdmg_base_keys.append(self._hdmg_base_key(key))
        n = len(self._hdmg_nodes)
        if n == 0:
            self._hdmg_built = False
            return
        distance_func = get_distance_function("cosine")
        rep_vectors = [self._hdmg_nodes[i][1].representative for i in range(n)]
        rep_vectors = [np.asarray(v).flatten() for v in rep_vectors]

        # Precompute the rep distance matrix and reuse it for subsequent embedding/semantic
        dist_matrix = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = distance_func(rep_vectors[i], rep_vectors[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Embedding edge: Each node is connected to the embedding_k nodes closest to rep (can span keys)
        directed_knn: List[List[int]] = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                d = dist_matrix[i, j]
                if embedding_max_distance is not None and d > embedding_max_distance:
                    continue
                dists.append((d, j))
            dists.sort(key=lambda x: x[0])
            directed_knn.append([j for _, j in dists[:embedding_k]])

        self._hdmg_embedding_edges = []
        if use_mutual_embedding:
            for i in range(n):
                mutual = [j for j in directed_knn[i] if i in directed_knn[j]]
                others = [j for j in directed_knn[i] if j not in mutual]
                self._hdmg_embedding_edges.append((mutual + others)[:embedding_k])
        else:
            self._hdmg_embedding_edges = directed_knn

        # Semantic edge: same-key neighbor + cross-key semantic bridge
        base_key_to_node_ids: Dict[str, List[int]] = {}
        for i in range(n):
            bk = self._hdmg_base_keys[i]
            if bk not in base_key_to_node_ids:
                base_key_to_node_ids[bk] = []
            base_key_to_node_ids[bk].append(i)
        key_vectors_available = len(self.key_vectors) > 0
        base_keys_list = list(self.key_vectors.keys()) if key_vectors_available else []

        # Each base_key selects a representative micro-cluster entry: select the node closest to the mean rep value in the key
        for bk, node_ids in base_key_to_node_ids.items():
            if not node_ids:
                continue
            if len(node_ids) == 1:
                self._hdmg_key_entries[bk] = node_ids[0]
                continue
            key_centroid = np.mean([rep_vectors[i] for i in node_ids], axis=0)
            best_entry = min(node_ids, key=lambda i: distance_func(key_centroid, rep_vectors[i]))
            self._hdmg_key_entries[bk] = best_entry

        # Pre-built entry matrix to speed up entry selection during query
        if len(self._hdmg_key_entries) > 0:
            sorted_items = sorted(self._hdmg_key_entries.items(), key=lambda x: x[0])
            self._hdmg_key_entry_base_keys = [bk for bk, _ in sorted_items]
            self._hdmg_key_entry_ids = [nid for _, nid in sorted_items]

            rep_rows = []
            key_rows = []
            valid_base_keys = []
            valid_node_ids = []
            for bk, nid in zip(self._hdmg_key_entry_base_keys, self._hdmg_key_entry_ids):
                rep = np.asarray(self._hdmg_nodes[nid][1].representative, dtype=np.float32).flatten()
                rep_norm = np.linalg.norm(rep)
                if np.isclose(rep_norm, 0.0):
                    rep_norm = 1.0
                rep_rows.append(rep / rep_norm)
                valid_base_keys.append(bk)
                valid_node_ids.append(nid)
                if bk in self.key_vectors:
                    kv = np.asarray(self.key_vectors[bk], dtype=np.float32).flatten()
                    kv_norm = np.linalg.norm(kv)
                    if np.isclose(kv_norm, 0.0):
                        kv_norm = 1.0
                    key_rows.append(kv / kv_norm)
                else:
                    key_rows.append(np.zeros_like(rep_rows[-1], dtype=np.float32))

            self._hdmg_key_entry_base_keys = valid_base_keys
            self._hdmg_key_entry_ids = valid_node_ids
            self._hdmg_key_entry_rep_matrix = np.vstack(rep_rows).astype(np.float32) if len(rep_rows) > 0 else None
            self._hdmg_key_entry_key_matrix = np.vstack(key_rows).astype(np.float32) if len(key_rows) > 0 else None

        self._hdmg_semantic_edges = []
        for i in range(n):
            neighbors: List[int] = []
            bk_i = self._hdmg_base_keys[i]
            same_key_ids = [j for j in base_key_to_node_ids.get(bk_i, []) if j != i]
            if same_key_ids:
                same_key_ids = sorted(same_key_ids, key=lambda j: dist_matrix[i, j])[:semantic_intra_k]
                neighbors.extend(same_key_ids)

            if key_vectors_available and bk_i in self.key_vectors:
                qk = np.asarray(self.key_vectors[bk_i]).flatten()
                other_base_keys = [b for b in base_keys_list if b != bk_i]
                if other_base_keys and len(rep_vectors) > 0:
                    other_key_dists = []
                    for b in other_base_keys:
                        kv = np.asarray(self.key_vectors[b]).flatten()
                        other_key_dists.append((distance_func(qk, kv), b))
                    other_key_dists.sort(key=lambda x: x[0])
                    for _, b in other_key_dists[:semantic_bridge_keys]:
                        cands = base_key_to_node_ids.get(b, [])
                        if cands:
                            sorted_cands = sorted(cands, key=lambda j: dist_matrix[i, j])[:semantic_bridge_per_key]
                            for cand in sorted_cands:
                                if cand not in neighbors:
                                    neighbors.append(cand)
            self._hdmg_semantic_edges.append(neighbors)

        # Global entry: rep is the node closest to the mean value of all reps (embedding entry)
        mean_rep = np.mean(rep_vectors, axis=0)
        entry_dists = [(distance_func(mean_rep, rep_vectors[i]), i) for i in range(n)]
        entry_dists.sort(key=lambda x: x[0])
        self._hdmg_global_entry = entry_dists[0][1]

        # Precompute the normalized representative matrix to speed up batch scoring during search
        rep_mat = np.vstack([np.asarray(v, dtype=np.float32) for v in rep_vectors])
        rn = np.linalg.norm(rep_mat, axis=1, keepdims=True)
        rn = np.where(rn == 0, 1.0, rn)
        self._hdmg_rep_matrix_normed = (rep_mat / rn).astype(np.float32)

        # Precompute node -> key index mapping + normalized key vector matrix
        unique_base_keys = sorted(set(self._hdmg_base_keys))
        self._hdmg_unique_base_keys = unique_base_keys
        key_to_idx = {k: i for i, k in enumerate(unique_base_keys)}
        self._hdmg_node_key_indices = np.array(
            [key_to_idx[self._hdmg_base_keys[i]] for i in range(n)], dtype=np.int32)
        key_vecs = []
        self._hdmg_key_has_vector = np.zeros(len(unique_base_keys), dtype=bool)
        for i, bk in enumerate(unique_base_keys):
            kv = self.key_vectors.get(bk)
            if kv is not None:
                kv = np.asarray(kv, dtype=np.float32).flatten()
                kn = np.linalg.norm(kv)
                if np.isclose(kn, 0.0):
                    kn = 1.0
                key_vecs.append(kv / kn)
                self._hdmg_key_has_vector[i] = True
            else:
                key_vecs.append(np.zeros(rep_mat.shape[1], dtype=np.float32))
        self._hdmg_key_matrix_normed = np.vstack(key_vecs).astype(np.float32)

        # Precompute the normalized vector matrix + meta-information of each node to speed up the rerank stage
        self._hdmg_node_vec_matrices = []  # [node_id] -> (M, D) normalized float32
        self._hdmg_node_vec_counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            key, group = self._hdmg_nodes[i]
            vecs = np.array([np.asarray(v, dtype=np.float32).flatten() for v in group.vectors])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            self._hdmg_node_vec_matrices.append((vecs / norms).astype(np.float32))
            self._hdmg_node_vec_counts[i] = len(group.vectors)

        self._hdmg_built = True

    def get_last_hdmg_search_stats(self) -> Dict[str, Any]:
        """HDMG 。"""
        return self._hdmg_last_search_stats

    def search_hdmg(self, query_vector: np.ndarray, query_key_vector: Optional[np.ndarray] = None,
                  alpha: float = 0.5, top_k: int = 10, max_steps: int = 100,
                  distance_method: str = "cosine", entry_alpha_threshold: float = 0.0,
                  cluster_pool_size: Optional[int] = None, top_key_candidates: int = 5,
                  extra_hops: int = 0) -> List[SearchResult]:
        """
        HDMG ：，， top_k。 alpha：， beta ； key。
        """
        mixed_score_calls = 0

        if cluster_pool_size is None:
            cluster_pool_size = top_k
        cluster_pool_size = max(1, cluster_pool_size)

        if not self._hdmg_built or len(self._hdmg_nodes) == 0:
            self._hdmg_last_search_stats = {
                "mode": "fallback",
                "reason": "hdmg_not_built_or_empty",
                "entries_count": 0,
                "route_depths": [],
                "route_hops": [],
                "candidate_pool_size": 0,
                "selected_nodes": 0,
            }
            if query_key_vector is not None and len(self.key_vectors) > 0:
                return self.search_with_mixed_key_rep_vec(
                    query_vector, query_key_vector=query_key_vector, beta=alpha, top_k=top_k,
                    gruop_expansion_factor=max(10, cluster_pool_size), distance_method=distance_method)
            return self.search_with_rep_vec(query_vector, top_k=top_k, gruop_expansion_factor=10,
                                           distance_method=distance_method)
        distance_func = get_distance_function(distance_method)
        q = np.asarray(query_vector).flatten()
        q_norm = np.linalg.norm(q)
        if np.isclose(q_norm, 0.0):
            q_norm = 1.0
        q_normed = (q / q_norm).astype(np.float32)
        n = len(self._hdmg_nodes)
        has_key = query_key_vector is not None and len(self.key_vectors) > 0
        q_key = np.asarray(query_key_vector).flatten() if has_key else None
        if q_key is not None:
            qk_norm = np.linalg.norm(q_key)
            if np.isclose(qk_norm, 0.0):
                qk_norm = 1.0
            q_key_normed = (q_key / qk_norm).astype(np.float32)
        else:
            q_key_normed = None

        import time as _time
        _t_entry_start = _time.perf_counter()

        def mixed_score(node_id: int) -> float:
            nonlocal mixed_score_calls
            mixed_score_calls += 1
            key, group = self._hdmg_nodes[node_id]
            rep = np.asarray(group.representative).flatten()
            emb_d = distance_func(q, rep)
            if has_key and q_key is not None:
                kv = self.get_key_vector(key)
                if kv is not None:
                    sem_d = distance_func(q_key, np.asarray(kv).flatten())
                    return alpha * sem_d + (1.0 - alpha) * emb_d
            return emb_d

        entries: List[int] = []
        if has_key and alpha > entry_alpha_threshold and q_key is not None:
            # Vectorized key entry screening: calculate the similarity on the key entry matrix at once
            if self._hdmg_key_entry_key_matrix is not None and q_key_normed is not None:
                sims = self._hdmg_key_entry_key_matrix @ q_key_normed
                k = min(max(1, top_key_candidates), len(sims))
                if k == len(sims):
                    top_idx = np.argsort(-sims)
                else:
                    part = np.argpartition(-sims, k - 1)[:k]
                    top_idx = part[np.argsort(-sims[part])]
                for idx in top_idx[:k]:
                    entries.append(self._hdmg_key_entry_ids[int(idx)])
            else:
                key_candidates: List[Tuple[float, str]] = []
                for bk, kv in self.key_vectors.items():
                    d = distance_func(q_key, np.asarray(kv).flatten())
                    key_candidates.append((d, bk))
                key_candidates.sort(key=lambda x: x[0])
                for _, cand_key in key_candidates[:max(1, top_key_candidates)]:
                    if cand_key in self._hdmg_key_entries:
                        entries.append(self._hdmg_key_entries[cand_key])
        else:
            # When alpha is small, press embedding to select top-K as the entrance from the representative entrance of each key.
            if self._hdmg_key_entry_rep_matrix is not None:
                sims = self._hdmg_key_entry_rep_matrix @ q_normed
                k = min(max(1, top_key_candidates), len(sims))
                if k == len(sims):
                    top_idx = np.argsort(-sims)
                else:
                    part = np.argpartition(-sims, k - 1)[:k]
                    top_idx = part[np.argsort(-sims[part])]
                for idx in top_idx[:k]:
                    entries.append(self._hdmg_key_entry_ids[int(idx)])
            else:
                for _, node_id in sorted(
                    ((mixed_score(node_id), node_id) for node_id in self._hdmg_key_entries.values()),
                    key=lambda x: x[0]
                )[:max(1, top_key_candidates)]:
                    entries.append(node_id)

        # Deduplication; if there is no key entry available, fall back to the global entry
        entries = list(dict.fromkeys(entries))
        if not entries and self._hdmg_global_entry is not None:
            entries = [self._hdmg_global_entry]
        _t_walk_start = _time.perf_counter()
        # Maintain candidate micro-cluster pool during the walk (node_id -> best mixed_score)
        candidate_nodes: Dict[int, float] = {}
        route_depths: List[int] = []
        route_hops: List[int] = []
        route_stop_reasons: List[str] = []
        node_score_cache: Dict[int, float] = {}

        def update_candidate(node_id: int, score: float) -> None:
            prev = candidate_nodes.get(node_id)
            if prev is None or score < prev:
                candidate_nodes[node_id] = score

        def get_mixed_score_cached(node_id: int) -> float:
            if node_id in node_score_cache:
                return node_score_cache[node_id]
            s = mixed_score(node_id)
            node_score_cache[node_id] = s
            return s

        # Precompute the semantic distance of all key pairs for query (only counted once for each search)
        _key_sem_dists = None
        if has_key and q_key_normed is not None and self._hdmg_key_matrix_normed is not None:
            _key_sims = self._hdmg_key_matrix_normed @ q_key_normed
            _key_sem_dists = (1.0 - _key_sims).astype(np.float32)

        def batch_mixed_score(node_ids: List[int]) -> Dict[int, float]:
            """mixed_score — ， + 。"""
            nonlocal mixed_score_calls
            if not node_ids:
                return {}
            if distance_method != "cosine":
                return {nid: get_mixed_score_cached(nid) for nid in node_ids}
            mixed_score_calls += len(node_ids)
            ids_arr = np.array(node_ids, dtype=np.intp)
            emb_sims = self._hdmg_rep_matrix_normed[ids_arr] @ q_normed
            emb_d = 1.0 - emb_sims
            if _key_sem_dists is not None:
                key_indices = self._hdmg_node_key_indices[ids_arr]
                sem_d = _key_sem_dists[key_indices]
                has_vec = self._hdmg_key_has_vector[key_indices]
                mixed = np.where(has_vec, alpha * sem_d + (1.0 - alpha) * emb_d, emb_d)
            else:
                mixed = emb_d
            return dict(zip(node_ids, mixed.tolist()))

        for entry in entries:
            current = entry
            current_score = get_mixed_score_cached(entry)
            update_candidate(current, current_score)

            steps = 0
            hops = 0
            stop_reason = "max_steps"
            while steps < max_steps:
                steps += 1
                all_neighbors = list(set(
                    self._hdmg_embedding_edges[current] + self._hdmg_semantic_edges[current]
                ))
                uncached = [j for j in all_neighbors if j not in node_score_cache]
                if uncached:
                    for nid, s in batch_mixed_score(uncached).items():
                        node_score_cache[nid] = s
                next_best = None
                next_best_score = current_score
                for j in all_neighbors:
                    s = node_score_cache[j]
                    update_candidate(j, s)
                    if s < next_best_score:
                        next_best_score = s
                        next_best = j
                if next_best is None:
                    stop_reason = "no_better_neighbor"
                    break
                current = next_best
                current_score = next_best_score
                hops += 1

            route_depths.append(steps)
            route_hops.append(hops)
            route_stop_reasons.append(stop_reason)

        # extra_hops: After greedy walk stops, expand N-hop neighbors from all nodes in the candidate pool (batched)
        for _hop in range(extra_hops):
            frontier = list(candidate_nodes.keys())
            new_neighbors = set()
            for nid in frontier:
                for j in self._hdmg_embedding_edges[nid] + self._hdmg_semantic_edges[nid]:
                    if j not in candidate_nodes:
                        new_neighbors.add(j)
            new_neighbors = list(new_neighbors)
            if new_neighbors:
                uncached_hop = [j for j in new_neighbors if j not in node_score_cache]
                if uncached_hop:
                    for nid, s in batch_mixed_score(uncached_hop).items():
                        node_score_cache[nid] = s
                for j in new_neighbors:
                    update_candidate(j, node_score_cache[j])

        _t_rerank_start = _time.perf_counter()
        selected_nodes = sorted(candidate_nodes.items(), key=lambda x: x[1])[:cluster_pool_size]

        all_vector_results: List[SearchResult] = []

        # Concatenate candidate vectors using a precomputed normalized matrix (avoid vector-by-vector np.asarray loop)
        use_precomputed = hasattr(self, '_hdmg_node_vec_matrices') and self._hdmg_node_vec_matrices
        if use_precomputed:
            mat_parts = []
            sem_d_parts = []
            node_meta = []  # (node_id, key, group, n_vecs)
            for node_id, _ in selected_nodes:
                key, group = self._hdmg_nodes[node_id]
                mat_parts.append(self._hdmg_node_vec_matrices[node_id])
                nv = self._hdmg_node_vec_counts[node_id]
                sem_d = None
                if _key_sem_dists is not None:
                    kidx = self._hdmg_node_key_indices[node_id]
                    if self._hdmg_key_has_vector[kidx]:
                        sem_d = float(_key_sem_dists[kidx])
                sem_d_parts.extend([sem_d] * nv)
                node_meta.append((node_id, key, group, int(nv)))

            if mat_parts:
                cand_matrix = np.vstack(mat_parts)
                emb_dists = 1.0 - (cand_matrix @ q_normed)

                sem_arr = np.array([(s if s is not None else np.nan) for s in sem_d_parts], dtype=np.float32)
                has_sem = ~np.isnan(sem_arr)
                scores = np.where(has_sem, alpha * sem_arr + (1.0 - alpha) * emb_dists, emb_dists)

                # Use argpartition to get top-k to avoid full sorting + avoid creating all SearchResults
                actual_k = min(top_k, len(scores))
                if actual_k < len(scores):
                    top_idx = np.argpartition(scores, actual_k - 1)[:actual_k]
                    top_idx = top_idx[np.argsort(scores[top_idx])]
                else:
                    top_idx = np.argsort(scores)

                # Create a mapping of flat index -> (key, group, vec_idx) and construct SearchResult only for top-k
                flat_node_id = np.empty(len(scores), dtype=np.int32)
                flat_vec_idx = np.empty(len(scores), dtype=np.int32)
                offset = 0
                for mi, (_, key, group, nv) in enumerate(node_meta):
                    flat_node_id[offset:offset+nv] = mi
                    flat_vec_idx[offset:offset+nv] = np.arange(nv)
                    offset += nv

                for rank_i in top_idx:
                    ri = int(rank_i)
                    mi = flat_node_id[ri]
                    _, rkey, rgroup, _ = node_meta[mi]
                    all_vector_results.append(SearchResult(float(scores[ri]), rkey, rgroup, int(flat_vec_idx[ri])))
        else:
            flat_vectors = []
            flat_keys: List[str] = []
            flat_groups: List[VectorGroup] = []
            flat_indices: List[int] = []
            flat_sem_d: List[Optional[float]] = []
            for node_id, _ in selected_nodes:
                key, group = self._hdmg_nodes[node_id]
                sem_d = None
                if _key_sem_dists is not None:
                    kidx = self._hdmg_node_key_indices[node_id]
                    if self._hdmg_key_has_vector[kidx]:
                        sem_d = float(_key_sem_dists[kidx])
                for idx, vec in enumerate(group.vectors):
                    flat_vectors.append(np.asarray(vec, dtype=np.float32).flatten())
                    flat_keys.append(key)
                    flat_groups.append(group)
                    flat_indices.append(idx)
                    flat_sem_d.append(sem_d)

            if len(flat_vectors) > 0:
                cand_matrix = np.vstack(flat_vectors).astype(np.float32)
                cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
                cand_norms = np.where(cand_norms == 0.0, 1.0, cand_norms)
                cand_matrix = cand_matrix / cand_norms
                emb_dists = 1.0 - (cand_matrix @ q_normed)
                for i in range(len(flat_vectors)):
                    emb_d = float(emb_dists[i])
                    sd = flat_sem_d[i]
                    score = alpha * sd + (1.0 - alpha) * emb_d if sd is not None else emb_d
                    all_vector_results.append(SearchResult(score, flat_keys[i], flat_groups[i], flat_indices[i]))
                all_vector_results.sort()

        _t_end = _time.perf_counter()
        self._hdmg_last_search_stats = {
            "mode": "hdmg",
            "entries_count": len(entries),
            "entry_node_ids": entries,
            "route_depths": route_depths,
            "route_hops": route_hops,
            "route_stop_reasons": route_stop_reasons,
            "candidate_pool_size": len(candidate_nodes),
            "selected_nodes": len(selected_nodes),
            "cluster_pool_size_limit": cluster_pool_size,
            "mixed_score_calls": mixed_score_calls,
            "t_entry_ms": (_t_walk_start - _t_entry_start) * 1000,
            "t_walk_ms": (_t_rerank_start - _t_walk_start) * 1000,
            "t_rerank_ms": (_t_end - _t_rerank_start) * 1000,
            "t_total_ms": (_t_end - _t_entry_start) * 1000,
        }

        return all_vector_results[:top_k]

    def search_with_contextual_vectors(self, query_vector: np.ndarray, top_k: int = 5, num: int = 2,
                                        distance_method: str = "cosine", weight_decay: float = 0.8) -> List[SearchResult]:
        """
        ， ： 1. search_with_rep_vec top_k * 2 2. （numnum） 3. （，） 4. ， top_k Args: query_vector: top_k: k num: ，num distance_method: weight_decay: ， Returns: List[SearchResult]:
        """
        # 1. First use search_with_rep_vec to find top_k * 2 results
        initial_results = self.search_with_rep_vec(query_vector, top_k=top_k * 2, distance_method=distance_method)

        if not initial_results:
            return []

        # Used to store the weighted score for each result
        scored_results = []
        distance_func = get_distance_function(distance_method)

        for result in initial_results:
            # 2. Get the context vector
            contextual_vectors = self.get_contextual_vectors(result, num=num)

            # 3. Calculate weighted distance
            # Find the position of the original result in the context (center position)
            center_idx = None
            for i, ctx_result in enumerate(contextual_vectors):
                if (ctx_result.group == result.group and
                    ctx_result.vector_idx == result.vector_idx):
                    center_idx = i
                    break

            if center_idx is None:
                center_idx = len(contextual_vectors) // 2

            # Calculate weighted distance score
            total_weighted_distance = 0.0
            total_weight = 0.0

            for i, ctx_result in enumerate(contextual_vectors):
                # Get context vector
                ctx_vector = ctx_result.group.vectors[ctx_result.vector_idx]

                # Calculate distance from query
                distance = distance_func(query_vector, ctx_vector)

                # Calculate the weight: the center weight is 1, and decreases toward both sides by weight_decay
                offset = abs(i - center_idx)
                weight = weight_decay ** offset

                total_weighted_distance += distance * weight
                total_weight += weight

            # Calculate the average weighted distance
            avg_weighted_distance = total_weighted_distance / total_weight if total_weight > 0 else float('inf')

            scored_results.append((avg_weighted_distance, result))

        # 4. Sort by weighted distance and return top_k results
        scored_results.sort(key=lambda x: x[0])

        # Update the distance of the result to the weighted distance and return top_k
        final_results = []
        for weighted_distance, result in scored_results[:top_k]:
            # Create a new SearchResult, using weighted distance
            final_results.append(SearchResult(
                distance=weighted_distance,
                group=result.group,
                vector_idx=result.vector_idx,
                key=result.key
            ))

        return final_results

    # By default, 2 are found before and after the context.
    def get_contextual_vectors(self, result: SearchResult, num: int=2) -> List[SearchResult]:
        # If vector_idx is None, it means that the result is a group level and the specific vector cannot be obtained.
        if result.vector_idx is None:
            return [result]

        cur_id = result.group.descriptions[result.vector_idx].get('id')
        if cur_id is None:
            return [result]

        # Parse id prefix and index
        id_parts = cur_id.rsplit('_', 1)
        if len(id_parts) != 2:
            return [result]

        id_prefix = id_parts[0]
        try:
            id_idx = int(id_parts[1])
        except ValueError:
            return [result]

        # Calculate the search range: num before and after
        start_idx = max(0, id_idx - num)
        end_idx = id_idx + num + 1

        contextual_results = []

        for i in range(start_idx, end_idx):
            target_id = f'{id_prefix}_{i}'

            if target_id not in self.context_map:
                continue

            # The format of context_map value: key%group_name%insert_idx
            mapped_value = self.context_map[target_id]
            parts = mapped_value.split('%')
            if len(parts) != 3:
                continue

            mapped_key = parts[0]
            mapped_group_name = parts[1]
            mapped_insert_idx = int(parts[2])

            mapped_group = self.get_group_by_name(mapped_key, mapped_group_name)
            if mapped_group is None:
                continue

            if mapped_insert_idx < 0 or mapped_insert_idx >= mapped_group.size():
                continue

            # Directly use insert_idx saved in context_map as vector index
            contextual_results.append(
                SearchResult(
                    distance=result.distance,
                    group=mapped_group,
                    vector_idx=mapped_insert_idx,
                    key=mapped_key
                )
            )

        # If no relevant vector is found, at least return the original result
        if not contextual_results:
            return [result]

        return contextual_results


    def get_paired_vectors(self, key: str, group_id: str) -> List[Tuple[VectorGroup, Dict]]:
        """
        Args: key: group_id: ID Returns: List[Tuple[VectorGroup, Dict]]:
        """
        paired_groups = []

        if key not in self.data:
            return paired_groups

        relations = self.get_relations(key)

        for pair_rel in relations.get('pair', []):
            other_group_id = None

            if pair_rel['group1_id'] == group_id:
                other_group_id = pair_rel['group2_id']
            elif pair_rel['group2_id'] == group_id:
                other_group_id = pair_rel['group1_id']

            if other_group_id:
                other_group = self.get_group_by_id(key, other_group_id)
                if other_group:
                    paired_groups.append((other_group, pair_rel))

        return paired_groups

    # ============= Search method based on correlation vector =============

    def get_vector_by_ref(self, ref: Union[VectorRef, str]) -> Optional[Tuple[np.ndarray, Dict, 'VectorGroup']]:
        """
        VectorRef 、description group Args: ref: VectorRef "key%group_name%vector_idx"

        Returns:
            (vector, description, group) None
        """
        if isinstance(ref, str):
            ref = VectorRef.from_string(ref)

        group = self.get_group_by_name(ref.key, ref.group_name)
        if group is None:
            return None

        if ref.vector_idx < 0 or ref.vector_idx >= len(group.vectors):
            return None

        return (group.vectors[ref.vector_idx],
                group.descriptions[ref.vector_idx],
                group)

    def get_related_vectors(self, result: SearchResult,
                            relation_type: str = None,
                            include_self: bool = True) -> List[Tuple[SearchResult, VectorRelation]]:
        """
        Args: result: relation_type: ， include_self: （ True） Returns: List[(SearchResult, VectorRelation)]:
        """
        related_results = []

        if result.vector_idx is None:
            return related_results

        # Get the description of the current vector
        desc = result.group.descriptions[result.vector_idx]

        # Contains itself
        if include_self:
            related_results.append((result, None))

        # Get associated vector
        relations = get_relations_from_description(desc, relation_type)

        for relation in relations:
            ref = relation.ref
            vec_data = self.get_vector_by_ref(ref)

            if vec_data is None:
                continue

            vector, related_desc, group = vec_data
            related_result = SearchResult(
                distance=result.distance,  # Temporarily use the distance from the original result
                key=ref.key,
                group=group,
                vector_idx=ref.vector_idx
            )
            related_results.append((related_result, relation))

        return related_results

    def search_with_relations(self,
                               query_vector: np.ndarray,
                               top_k: int = 5,
                               relation_types: List[str] = None,
                               relation_weight_factor: float = 0.5,
                               distance_method: str = "cosine",
                               print_details: bool = False) -> List[SearchResult]:
        """
        ， description related_vectors ： 1. search_with_rep_vec top_k * 2 2. ， 3. ：original_score + relation_weight_factor * weighted_relation_score
        4. top_k Args: query_vector: top_k: k relation_types: （None ） relation_weight_factor: distance_method: print_details: Returns: List[SearchResult]:
        """
        # 1. Get preliminary results
        initial_results = self.search_with_rep_vec(
            query_vector,
            top_k=top_k * 2,
            distance_method=distance_method
        )

        if not initial_results:
            return []

        distance_func = get_distance_function(distance_method)
        scored_results = []

        for result in initial_results:
            if result.vector_idx is None:
                scored_results.append((result.distance, result))
                continue

            # 2. Get the associated vector
            related_items = self.get_related_vectors(result, include_self=False)

            if not related_items:
                # No associated vectors, use raw distance
                scored_results.append((result.distance, result))
                continue

            # 3. Calculate the weighted score of the association vector
            total_relation_score = 0.0
            total_weight = 0.0

            for related_result, relation in related_items:
                if relation is None:
                    continue

                # Filter relationship types
                if relation_types and relation.relation_type not in relation_types:
                    continue

                # Get associated vector
                vec_data = self.get_vector_by_ref(relation.ref)
                if vec_data is None:
                    continue

                related_vector, _, _ = vec_data

                # Calculate the distance between the association vector and the query vector
                related_distance = distance_func(query_vector, related_vector)

                # weighted accumulation
                weight = relation.weight
                total_relation_score += related_distance * weight
                total_weight += weight

            # Calculate overall score
            if total_weight > 0:
                avg_relation_score = total_relation_score / total_weight
                final_score = (1 - relation_weight_factor) * result.distance +\
                              relation_weight_factor * avg_relation_score
            else:
                final_score = result.distance

            scored_results.append((final_score, result))

        # 4. Sort by overall score
        scored_results.sort(key=lambda x: x[0])

        # Print details
        if print_details:
            print(f"\n{'='*60}")
            print(f"search_with_relations (relation_weight_factor={relation_weight_factor})")
            print(f"{'='*60}")
            for i, (score, result) in enumerate(scored_results[:top_k]):
                desc = result.group.descriptions[result.vector_idx] if result.vector_idx is not None else {}
                related_count = len(desc.get("related_vectors", []))
                print(f"  [{i+1}] key={result.key}, group={result.group.group_name}, "
                      f"idx={result.vector_idx}, final_score={score:.6f}, "
                      f"original_dist={result.distance:.6f}, related_count={related_count}")

        # Construct the final result (update distance to the combined score)
        final_results = []
        for score, result in scored_results[:top_k]:
            final_results.append(SearchResult(
                distance=score,
                key=result.key,
                group=result.group,
                vector_idx=result.vector_idx
            ))

        return final_results

    def search_multimodal(self,
                          query_vectors: Dict[str, np.ndarray],
                          modality_weights: Dict[str, float] = None,
                          top_k: int = 5,
                          distance_method: str = "cosine",
                          print_details: bool = False) -> List[SearchResult]:
        """
        ：（+）， Args: query_vectors: ， {"image": img_vec, "text": text_vec}
            modality_weights: ， {"image": 0.6, "text": 0.4}
            top_k: k distance_method: print_details: Returns: List[SearchResult]:
        """
        if not query_vectors:
            return []

        # Default weight: equally distributed
        if modality_weights is None:
            n_modalities = len(query_vectors)
            modality_weights = {k: 1.0 / n_modalities for k in query_vectors}

        # Search for each modality separately
        modality_results: Dict[str, List[SearchResult]] = {}
        for modality, query_vec in query_vectors.items():
            results = self.search_with_rep_vec(
                query_vec,
                top_k=top_k * 3,  # Take more for fusion
                distance_method=distance_method
            )
            modality_results[modality] = results

        # Collect all candidate results and calculate the fusion score
        candidate_scores: Dict[str, Tuple[float, SearchResult]] = {}  # key: "key%group%idx" -> (score, result)
        distance_func = get_distance_function(distance_method)

        for modality, results in modality_results.items():
            weight = modality_weights.get(modality, 1.0)
            query_vec = query_vectors[modality]

            for result in results:
                if result.vector_idx is None:
                    continue

                # Generate unique identifier
                result_key = f"{result.key}%{result.group.group_name}%{result.vector_idx}"

                # Calculate the score for this result
                score = result.distance * weight

                # Check association vectors to see if there are cross-modal associations
                desc = result.group.descriptions[result.vector_idx]
                relations = get_relations_from_description(desc)

                for relation in relations:
                    # Get associated vector
                    vec_data = self.get_vector_by_ref(relation.ref)
                    if vec_data is None:
                        continue

                    related_vec, related_desc, _ = vec_data

                    # Calculate the distance from the query vector to the associated vector for each modality
                    for other_modality, other_query_vec in query_vectors.items():
                        if other_modality == modality:
                            continue

                        other_weight = modality_weights.get(other_modality, 1.0)
                        related_dist = distance_func(other_query_vec, related_vec)

                        # Plus cross-modal correlation score
                        cross_modal_score = related_dist * other_weight * relation.weight * 0.5
                        score += cross_modal_score

                # Accumulate or update scores
                if result_key in candidate_scores:
                    existing_score, existing_result = candidate_scores[result_key]
                    candidate_scores[result_key] = (existing_score + score, existing_result)
                else:
                    candidate_scores[result_key] = (score, result)

        # Sort and return top_k
        sorted_candidates = sorted(candidate_scores.values(), key=lambda x: x[0])

        if print_details:
            print(f"\n{'='*60}")
            print("search_multimodal results")
            print(f"modality_weights: {modality_weights}")
            print(f"{'='*60}")
            for i, (score, result) in enumerate(sorted_candidates[:top_k]):
                desc = result.group.descriptions[result.vector_idx] if result.vector_idx is not None else {}
                print(f"  [{i+1}] key={result.key}, group={result.group.group_name}, "
                      f"idx={result.vector_idx}, fused_score={score:.6f}")

        final_results = []
        for score, result in sorted_candidates[:top_k]:
            final_results.append(SearchResult(
                distance=score,
                key=result.key,
                group=result.group,
                vector_idx=result.vector_idx
            ))

        return final_results

    # ============= Statistics =============

    def get_statistics(self) -> Dict[str, Any]:
        """Return the requested VectorMap data."""
        stats = {
            'total_keys': len(self.data),
            'total_groups': 0,
            'total_vectors': 0,
            'vector_types': set(),
            'group_types': set(),
            'total_pair_relations': 0,
            'total_tree_relations': 0,
            'total_custom_relations': 0,
            'key_details': {},
        }

        for key, entry in self.data.items():
            groups = entry['groups']
            stats['total_groups'] += len(groups)

            key_vectors = sum(group.size() for group in groups)
            stats['total_vectors'] += key_vectors

            # Collection type information
            for group in groups:
                stats['vector_types'].add(group.vector_type)
                stats['group_types'].add(group.group_type)

            # Number of statistical relationships
            relations = entry['metadata'].get('relations', {})
            stats['total_pair_relations'] += len(relations.get('pair', []))
            stats['total_tree_relations'] += len(relations.get('tree', []))
            stats['total_custom_relations'] += len(relations.get('custom', []))

            # key details
            stats['key_details'][key] = {
                'groups_count': len(groups),
                'vectors_count': key_vectors,
                'pair_relations': len(relations.get('pair', [])),
                'tree_relations': len(relations.get('tree', [])),
                'custom_relations': len(relations.get('custom', [])),
                'vector_types': list(entry['metadata'].get('vector_types', set())),
                'group_types': list(entry['metadata'].get('group_types', set())),
            }

        # Convert collection to list
        stats['vector_types'] = list(stats['vector_types'])
        stats['group_types'] = list(stats['group_types'])

        return stats

    def analyze_relationships(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Return the requested VectorMap data."""
        analysis = {
            'pair_analysis': {
                'total_pairs': 0,
                'relation_types': set(),
                'avg_vector_count': 0
            },
            'tree_analysis': {
                'total_nodes': 0,
                'max_depth': 0,
                'root_nodes': set()
            },
            'connectivity': {
                'connected_groups': set(),
                'isolated_groups': set()
            }
        }

        search_keys = [key] if key else list(self.data.keys())

        for k in search_keys:
            if k not in self.data:
                continue

            relations = self.get_relations(k)

            # Analyze pair relationships
            for pair_rel in relations.get('pair', []):
                analysis['pair_analysis']['total_pairs'] += 1
                analysis['pair_analysis']['relation_types'].add(pair_rel.get('relation_type', 'unknown'))
                analysis['connectivity']['connected_groups'].add(pair_rel['group1_id'])
                analysis['connectivity']['connected_groups'].add(pair_rel['group2_id'])

            # Analyze tree relationships
            hierarchy = self.get_hierarchy_tree(k)
            analysis['tree_analysis']['root_nodes'].update(hierarchy.get('root_nodes', []))

            # Calculate tree depth (simple implementation)
            def calculate_depth(node, parent_children, current_depth=0):
                children = parent_children.get(node, [])
                if not children:
                    return current_depth
                return max(calculate_depth(child, parent_children, current_depth + 1)
                          for child in children)

            for root in hierarchy.get('root_nodes', []):
                depth = calculate_depth(root, hierarchy.get('parent_children', {}))
                analysis['tree_analysis']['max_depth'] = max(analysis['tree_analysis']['max_depth'], depth)

        # Convert collection to list
        analysis['pair_analysis']['relation_types'] = list(analysis['pair_analysis']['relation_types'])
        analysis['tree_analysis']['root_nodes'] = list(analysis['tree_analysis']['root_nodes'])
        analysis['connectivity']['connected_groups'] = list(analysis['connectivity']['connected_groups'])
        analysis['connectivity']['isolated_groups'] = list(analysis['connectivity']['isolated_groups'])

        return analysis

    def analyze_vectormap_storage(self) -> None:
        """VectorMap"""

        print("\n=== VectorMap Storage Analysis ===")
        total_groups = len(self.data)
        total_vectors = 0
        total_descriptions = 0

        print(f"Total keys: {total_groups}")
        print(f"Keys: {list(self.data.keys())}")

        print(f"\n:")
        print(f"{'Key':<15} {'Group':<30} {'Vectors':<8} {'Dimension':<10} {'Type':<10}")
        print("-" * 80)

        for key, data in self.data.items():
            groups = data['groups']
            for group in groups:
                vector_count = len(group.vectors)
                vector_dim = len(group.representative)
                total_vectors += vector_count
                total_descriptions += len(group.descriptions)

                print(f"{key:<15} {group.group_name:<30} {vector_count:<8} {vector_dim:<10} {group.vector_type:<10}")

        print(f"\n:")
        print(f"  : {total_groups}")
        print(f"  Total number of vectors: {total_vectors}")
        print(f"  : {total_descriptions}")
        print(f"  : {total_vectors / total_groups:.2f}")

        return {
            "total_groups": total_groups,
            "total_vectors": total_vectors,
            "total_descriptions": total_descriptions,
            "avg_vectors_per_group": total_vectors / total_groups
        }

    def print_hierarchy_tree(self, key: str, show_group_info: bool = True) -> None:
        """
        key Args: key: show_group_info:
        """
        hierarchy = self.get_hierarchy_tree(key)
        if not hierarchy['root_nodes']:
            print("No hierarchy data found")
            return

        print(f"--- {key} ---")
        self._print_tree_structure(
            hierarchy['parent_children'],
            hierarchy['root_nodes'],
            hierarchy['node_groups'] if show_group_info else {},
            show_group_info
        )

    def _print_tree_structure(self, parent_children: Dict, root_nodes: List[str],
                             node_groups: Dict, show_group_info: bool,
                             prefix: str = "", is_last: bool = True) -> None:
        """Return the requested VectorMap data."""
        for i, root in enumerate(root_nodes):
            is_last_node = i == len(root_nodes) - 1
            current_prefix = prefix + ("└── " if is_last_node else "├── ")

            # Print node name and vector group information
            if show_group_info and node_groups:
                group_ids = node_groups.get(root, [])
                group_info = f" (vector groups: {len(group_ids)})" if group_ids else " (no vector group)"
            else:
                group_info = ""

            print(f"{current_prefix}{root}{group_info}")

            # Recursively print child nodes
            children = parent_children.get(root, [])
            if children:
                child_prefix = prefix + ("    " if is_last_node else "│   ")
                self._print_tree_structure(parent_children, children, node_groups,
                                         show_group_info, child_prefix, False)

    # ============= Other methods =============

    def __len__(self) -> int:
        """Return all stored keys."""
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        """Return all stored keys."""
        return key in self.data

    def __getitem__(self, key: str) -> Dict:
        """Return the requested VectorMap data."""
        return self.data[key]

    def __str__(self) -> str:
        """Return the requested VectorMap data."""
        return f"VectorMap(keys={len(self.data)}, groups={sum(len(entry['groups']) for entry in self.data.values())})"




if __name__ == "__main__":
    # Unit testing
    vectormap = VectorMap()

    # ========== Create test data structure ==========
    print("\n" + "=" * 60, flush=True)
    print("Create test data structure", flush=True)
    print("=" * 60, flush=True)

    np.random.seed(42)
    vector_dim = 128

    # Create 4 keys: anchor, puma, bike, soccer
    # puma has two groups: puma_face, puma_lateral
    # The other three keys each have a group with the same name: anchor, bike, soccer.

    # 1. anchor key with anchor group
    anchor_vector = np.random.rand(vector_dim)
    anchor_group = VectorGroup(
        group_name="anchor",
        representative=anchor_vector.copy(),
        rep_description="anchor representative vector",
        vectors=[anchor_vector],
        descriptions=[{"text": "anchor initial vector"}],
        vector_type="text",
        group_type="default"
    )
    vectormap.insert("anchor", anchor_group, metadata={"data_type": "text"})
    print(f"✅ key: anchor, group: anchor", flush=True)

    # 2. puma key with puma_face and puma_lateral groups
    puma_face_vector = np.random.rand(vector_dim)
    puma_face_group = VectorGroup(
        group_name="puma_face",
        representative=puma_face_vector.copy(),
        rep_description="puma face representative vector",
        vectors=[puma_face_vector],
        descriptions=[{"text": "puma face initial vector"}],
        vector_type="image",
        group_type="default"
    )
    vectormap.insert("puma", puma_face_group, metadata={"data_type": "image"})
    print(f"✅ key: puma, group: puma_face", flush=True)

    puma_lateral_vector = np.random.rand(vector_dim)
    puma_lateral_group = VectorGroup(
        group_name="puma_lateral",
        representative=puma_lateral_vector.copy(),
        rep_description="puma lateral representative vector",
        vectors=[puma_lateral_vector],
        descriptions=[{"text": "puma lateral initial vector"}],
        vector_type="image",
        group_type="default"
    )
    vectormap.insert("puma", puma_lateral_group, metadata={"data_type": "image"})
    print(f"✅ key: puma, group: puma_lateral", flush=True)

    # 3. bike key with bike group
    bike_vector = np.random.rand(vector_dim)
    bike_group = VectorGroup(
        group_name="bike",
        representative=bike_vector.copy(),
        rep_description="bike representative vector",
        vectors=[bike_vector],
        descriptions=[{"text": "bike initial vector"}],
        vector_type="text",
        group_type="default"
    )
    vectormap.insert("bike", bike_group, metadata={"data_type": "text"})
    print(f"✅ key: bike, group: bike", flush=True)

    # 4. soccer key with soccer group
    soccer_vector = np.random.rand(vector_dim)
    soccer_group = VectorGroup(
        group_name="soccer",
        representative=soccer_vector.copy(),
        rep_description="soccer representative vector",
        vectors=[soccer_vector],
        descriptions=[{"text": "soccer initial vector"}],
        vector_type="text",
        group_type="default"
    )
    vectormap.insert("soccer", soccer_group, metadata={"data_type": "text"})
    print(f"✅ key: soccer, group: soccer", flush=True)

    # Print statistics
    print(f"\n📊 VectorMap :", flush=True)
    print(f"   Key: {len(vectormap)}", flush=True)
    for key in vectormap.data.keys():
        groups = vectormap.data[key]['groups']
        print(f"   Key '{key}': {len(groups)} group", flush=True)
        for group in groups:
            print(f"      - group_name: {group.group_name}, size: {group.size()}", flush=True)

    # ========== Testing add_vector and add_vector_list ==========
    print("\n" + "=" * 60, flush=True)
    print("add_vector add_vector_list", flush=True)
    print("=" * 60, flush=True)

    # ========== Test add_vector_list - multiple sequences randomly assigned to different groups ==========
    print("\n" + "=" * 60, flush=True)
    print("add_vector_list -", flush=True)
    print("=" * 60, flush=True)

    # Define available key and group mappings
    keys = {"anchor": ["anchor"], "puma": ["puma_face", "puma_lateral"], "bike": ["bike"], "soccer": ["soccer"]}

    import random
    random.seed(42)  # Fixed random seed for easy reproduction, comment out to make it random

    # Record the number of vectors for each group before insertion
    group_sizes_before = {}
    for key, groups in keys.items():
        for group_name in groups:
            group = vectormap.get_group_by_name(key, group_name)
            if group:
                group_sizes_before[f"{key}_{group_name}"] = group.size()

    # Create multiple sequences
    num_sequences = 5  # Create 5 sequences
    sequence_lengths = [8, 12, 10, 15, 9]  # length of each sequence

    total_inserted_all = 0

    for seq_idx in range(num_sequences):
        sequence_length = sequence_lengths[seq_idx]

        print(f"\n{'=' * 60}", flush=True)
        print(f"Sequence {seq_idx + 1}/{num_sequences} (length: {sequence_length})", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Create a vector sequence (simulating a long text being divided into multiple segments)
        vector_list = [np.random.rand(vector_dim) for _ in range(sequence_length)]

        # Randomly select a key and group_name for each vector
        description_list = []
        print(f"\nGenerated {sequence_length} vectors and assigned them to random groups:", flush=True)

        for i in range(sequence_length):
            # Randomly select a key
            key = random.choice(list(keys.keys()))
            # Randomly select one from the groups corresponding to the key
            group_name = random.choice(keys[key])

            description_list.append({
                "key": key,
                "group_name": group_name,
                "text": f"sequence_{seq_idx + 1}_vector_{i + 1}",
                "uuid_idx": i
            })
            print(f"   [{i+1}]: key={key}, group_name={group_name}", flush=True)

        # Call add_vector_list to insert in batches
        print(f"\nBatch inserting sequence {seq_idx + 1}...", flush=True)
        result = vectormap.add_vector_list(vector_list, description_list)
        print(f"Sequence {seq_idx + 1} inserted {result} vectors", flush=True)
        total_inserted_all += result

    # Check the final insertion of each group
    print(f"\n{'=' * 60}", flush=True)
    print("📊 ，:", flush=True)
    print(f"{'=' * 60}", flush=True)

    total_inserted = 0
    for key, groups in keys.items():
        for group_name in groups:
            group = vectormap.get_group_by_name(key, group_name)
            if group:
                key_group = f"{key}_{group_name}"
                size_before = group_sizes_before.get(key_group, 0)
                size_after = group.size()
                inserted = size_after - size_before
                total_inserted += inserted
                print(f"   {key_group}: {size_before} -> {size_after} (inserted {inserted})", flush=True)

    print(f"\n   Total inserted: {total_inserted} vectors across {num_sequences} sequences", flush=True)

    # ========== Check context_map mapping ==========
    print(f"\n{'=' * 60}", flush=True)
    print("context_map", flush=True)
    print(f"{'=' * 60}", flush=True)

    print(f"   context_map : {len(vectormap.context_map)}", flush=True)
    print(f"   context_map （10）:", flush=True)
    for idx, (cur_id, mapped_value) in enumerate(vectormap.context_map.items()):
        print(f"      {cur_id} -> {mapped_value}", flush=True)


    print()
    #     if idx < 10:
    #         print(f"      {cur_id} -> {mapped_value}", flush=True)
    # if len(vectormap.context_map) > 10:
    # print(f" ... (and {len(vectormap.context_map) - 10} items)", flush=True)


    query_vector = np.random.rand(vector_dim)
    result = vectormap.search(query_vector, top_k=3)
    rep_results = vectormap.search_with_contextual_vectors(query_vector, top_k=3)

    # Compare two search results
    print(f"\n{'='*60}")
    print("search vs search_with_contextual_vectors top3")
    print(f"{'='*60}")

    print("\n[search ]:")
    for i, r in enumerate(result):
        desc = r.group.descriptions[r.vector_idx] if r.vector_idx is not None else {}
        print(f"  [{i+1}] key={r.key}, group={r.group.group_name}, idx={r.vector_idx}, "
              f"distance={r.distance:.6f}, id={desc.get('id', 'N/A')}")

    print("\n[search_with_contextual_vectors ]:")
    for i, r in enumerate(rep_results):
        desc = r.group.descriptions[r.vector_idx] if r.vector_idx is not None else {}
        print(f"  [{i+1}] key={r.key}, group={r.group.group_name}, idx={r.vector_idx}, "
              f"distance={r.distance:.6f}, id={desc.get('id', 'N/A')}")

    # Check if they are the same
    is_same = True
    if len(result) != len(rep_results):
        is_same = False
    else:
        for r1, r2 in zip(result, rep_results):
            if (r1.key != r2.key or
                r1.group.group_name != r2.group.group_name or
                r1.vector_idx != r2.vector_idx):
                is_same = False
                break

    print(f"\nConclusion: {'results are identical' if is_same else 'results differ'}")
    print(f"{'='*60}\n")

    # result = vectormap.search(query_vector, top_k=1)[0]
    # print(f"query result: {result}", flush=True)
    # print(result.key)
    # contextual_results = vectormap.get_contextual_vectors(result)
    # for idx, contextual_result in enumerate(contextual_results):
    # print(f"Context vector {idx}: {contextual_result}", flush=True)
