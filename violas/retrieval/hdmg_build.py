"""Batch cosine HDMG construction and changed-representative maintenance.

Preserves the original embedding-neighbor and semantic-bridge definitions.
"""
from collections import defaultdict
import numpy as np


def unit(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    norms=np.linalg.norm(matrix,axis=-1,keepdims=True)
    return matrix/np.where(norms==0,1,norms)


def build_fast_hdmg(self,embedding_k=16,semantic_intra_k=4,semantic_bridge_keys=2,
                    semantic_bridge_per_key=1,use_mutual_embedding=True,
                    embedding_max_distance=None,semantic_same_key_limit=None):
    if semantic_same_key_limit is not None:
        semantic_intra_k=semantic_same_key_limit
    nodes=[(key,g) for key,e in self.data.items() for g in e['groups'] if g.representative is not None]
    n=len(nodes)
    self._hdmg_nodes=nodes
    self._hdmg_base_keys=[self._hdmg_base_key(key) for key,_ in nodes]
    if not n:
        self._hdmg_built=False
        return
    reps=np.stack([np.asarray(g.representative).reshape(-1) for _,g in nodes]).astype(np.float64)
    normalized=unit(reps)
    identities=[(key,g.group_name) for key,g in nodes]
    previous=getattr(self,'_fast_hdmg_state',None)
    if previous is not None and previous['identities']==identities:
        changed=np.flatnonzero(np.any(reps!=previous['reps'],axis=1))
        distances=previous['distances']
        if len(changed):
            block=1-normalized[changed]@normalized.T
            distances[changed,:]=block
            distances[:,changed]=block.T
        affected=np.zeros(n,bool)
        affected[changed]=True
        if len(changed):
            # Vectorized dependency checks for fixed-degree neighbor lists.
            lengths=[len(row) for row in previous['directed']]
            if len(set(lengths))==1 and lengths[0]:
                old=np.asarray(previous['directed'],np.int32)
                affected|=np.isin(old,changed).any(axis=1)
                affected|=np.min(distances[:,changed],axis=1)<=np.max(distances[np.arange(n)[:,None],old],axis=1)
            else:
                for i,old in enumerate(previous['directed']):
                    if any(j in old for j in changed) or len(old)<min(embedding_k,n-1):affected[i]=True
                    if old and np.min(distances[i,changed])<=max(distances[i,old]):affected[i]=True
        directed=[list(v) for v in previous['directed']]
    else:
        distances=1-normalized@normalized.T
        affected=np.ones(n,bool)
        changed=np.arange(n)
        directed=[[] for _ in range(n)]
    np.fill_diagonal(distances,np.inf)
    for i in np.flatnonzero(affected):
        order=np.argsort(distances[i],kind='stable')
        order=order[order!=i]
        if embedding_max_distance is not None:
            order=order[distances[i,order]<=embedding_max_distance]
        directed[i]=order[:embedding_k].tolist()
    sets=[set(v) for v in directed]
    self._hdmg_embedding_edges=[([j for j in row if i in sets[j]]+[j for j in row if i not in sets[j]])
                               if use_mutual_embedding else row for i,row in enumerate(directed)]
    by_key=defaultdict(list)
    for i,key in enumerate(self._hdmg_base_keys):
        by_key[key].append(i)
    key_names=list(self.key_vectors)
    key_nearest={}
    if key_names:
        keys=unit(np.stack([np.asarray(self.key_vectors[k]).reshape(-1) for k in key_names]))
        kd=1-keys@keys.T
        np.fill_diagonal(kd,np.inf)
        key_nearest={key:[key_names[j] for j in np.argsort(kd[i],kind='stable')[:semantic_bridge_keys] if j!=i]
                     for i,key in enumerate(key_names)}
    incremental=previous is not None and previous['identities']==identities
    old_semantic=self._hdmg_semantic_edges if incremental else None
    self._hdmg_semantic_edges=[]
    old_entries=dict(self._hdmg_key_entries) if incremental else {}
    self._hdmg_key_entries={}
    changed_keys={self._hdmg_base_keys[i] for i in changed}
    for key,ids in by_key.items():
        if incremental and key not in changed_keys:
            self._hdmg_key_entries[key]=old_entries[key]
            continue
        center=unit(reps[ids].mean(axis=0))
        self._hdmg_key_entries[key]=ids[int(np.argmax(normalized[ids]@center))]
    for i,key in enumerate(self._hdmg_base_keys):
        if incremental and i not in changed and key not in changed_keys and not (set(key_nearest.get(key,[]))&changed_keys):
            self._hdmg_semantic_edges.append(old_semantic[i])
            continue
        same=np.asarray([j for j in by_key[key] if j!=i],dtype=int)
        edge=same[np.argsort(distances[i,same],kind='stable')[:semantic_intra_k]].tolist()
        for other in key_nearest.get(key,[]):
            candidates=np.asarray(by_key.get(other,[]),dtype=int)
            edge.extend(candidates[np.argsort(distances[i,candidates],kind='stable')[:semantic_bridge_per_key]].tolist())
        self._hdmg_semantic_edges.append(list(dict.fromkeys(edge)))
    self._hdmg_global_entry=int(np.argmax(normalized@unit(reps.mean(axis=0))))
    self._hdmg_rep_matrix_normed=normalized.astype(np.float32)
    base_keys=sorted(by_key)
    self._hdmg_unique_base_keys=base_keys
    key_ids={key:i for i,key in enumerate(base_keys)}
    self._hdmg_node_key_indices=np.array([key_ids[k] for k in self._hdmg_base_keys],np.int32)
    self._hdmg_key_has_vector=np.array([k in self.key_vectors for k in base_keys])
    self._hdmg_key_matrix_normed=unit(np.stack([np.asarray(self.key_vectors[k]).reshape(-1)
        if k in self.key_vectors else np.zeros(reps.shape[1]) for k in base_keys])).astype(np.float32)
    self._hdmg_key_entry_base_keys=base_keys
    self._hdmg_key_entry_ids=[self._hdmg_key_entries[k] for k in base_keys]
    self._hdmg_key_entry_key_matrix=self._hdmg_key_matrix_normed
    self._hdmg_key_entry_rep_matrix=self._hdmg_rep_matrix_normed[self._hdmg_key_entry_ids]
    matrices=getattr(self,'_hdmg_node_vec_matrices',[])
    if previous is None or previous['identities']!=identities or len(matrices)!=n:
        matrices=[None]*n
        changed=np.arange(n)
    # Also refresh changed member counts even if the mean did not move.
    counts=np.array([len(g.vectors) for _,g in nodes],np.int32)
    old_counts=getattr(self,'_hdmg_node_vec_counts',np.zeros(n,np.int32))
    touched=set(changed.tolist())
    if len(old_counts)==n:
        touched.update(np.flatnonzero(old_counts!=counts).tolist())
    touched.update(getattr(self,'_fast_member_dirty',set()))
    for i in touched:
        matrices[i]=(unit(np.stack(nodes[i][1].vectors)).astype(np.float32) if nodes[i][1].vectors
                     else np.empty((0,reps.shape[1]),np.float32))
    self._hdmg_node_vec_matrices=matrices
    self._hdmg_node_vec_counts=counts
    self._fast_member_dirty=set()
    self._fast_hdmg_state={'identities':identities,'reps':reps,'distances':distances,'directed':directed}
    self._hdmg_built=True
    self._fast_hdmg_stats={'changed_centers':len(changed),'neighbor_rows_updated':int(affected.sum()),'nodes':n}
