"""Update fixed-topology HDMG index state for changed member groups."""
from collections import defaultdict
import numpy as np
from .hdmg_build import unit


def prepare(vm,embedding_k=16,semantic_intra_k=8,bridge_keys=2,bridge_per_key=1):
    state=vm._fast_hdmg_state
    by_key=defaultdict(list)
    for i,key in enumerate(vm._hdmg_base_keys):by_key[key].append(i)
    keys=list(vm.key_vectors)
    matrix=unit(np.stack([np.asarray(vm.key_vectors[k]).reshape(-1) for k in keys]))
    distances=1-matrix@matrix.T
    np.fill_diagonal(distances,np.inf)
    state['normal64']=unit(state['reps'])
    state['directed_sets']=[set(row) for row in state['directed']]
    state['hot']=dict(by_key={k:np.asarray(v,int) for k,v in by_key.items()},
        key_nearest={k:[keys[j] for j in np.argsort(distances[i],kind='stable')[:bridge_keys] if j!=i] for i,k in enumerate(keys)},
        embedding_k=embedding_k,intra=semantic_intra_k,bridge_per_key=bridge_per_key)
    parts={}
    for key,rows in state['hot']['by_key'].items():
        parts[key]=[]
        for target in [key]+state['hot']['key_nearest'][key]:
            candidates=state['hot']['by_key'].get(target,np.empty(0,int))
            count=min(semantic_intra_k if target==key else bridge_per_key,
                      max(0,len(candidates)-(target==key)))
            order=np.argsort(state['distances'][np.ix_(rows,candidates)],axis=1,kind='stable')
            nearest=np.asarray([candidates[o[candidates[o]!=i][:count]] for i,o in zip(rows,order)],int)
            parts[key].append(dict(candidates=candidates,nearest=nearest,count=count))
    state['hot']['semantic_parts']=parts


def update(vm,dirty):
    if 'semantic_parts' not in vm._fast_hdmg_state.get('hot',{}):prepare(vm)
    state=vm._fast_hdmg_state
    config=state['hot'];by_key=config['by_key'];key_nearest=config['key_nearest']
    dirty=np.asarray(sorted(dirty),int)
    reps=state['reps'];n=len(reps)
    dirty_mask=np.zeros(n,bool)
    dirty_mask[dirty]=True
    for i in dirty:reps[i]=np.asarray(vm._hdmg_nodes[i][1].representative).reshape(-1)
    vm._hdmg_rep_matrix_normed[dirty]=unit(reps[dirty]).astype(np.float32)
    dist=state['distances']
    # Use float64 cached normals, avoiding float32 drift relative to full build.
    if 'normal64' not in state:state['normal64']=unit(reps)
    state['normal64'][dirty]=unit(reps[dirty])
    norm=state['normal64']
    block=1-norm[dirty]@norm.T
    dist[dirty,:]=block;dist[:,dirty]=block.T
    np.fill_diagonal(dist,np.inf)
    old=np.asarray(state['directed'],int)
    affected=dirty_mask[old].any(axis=1)
    affected|=np.min(dist[:,dirty],axis=1)<=np.max(dist[np.arange(n)[:,None],old],axis=1)
    affected[dirty]=True
    changes=np.flatnonzero(affected)
    orders=np.argsort(dist[changes],axis=1,kind='stable')
    for i,order in zip(changes,orders):
        state['directed'][i]=order[order!=i][:config['embedding_k']].tolist()
    directed=state['directed']
    edge_rows=affected | affected[old].any(axis=1)
    sets=state['directed_sets']
    for i in changes:sets[i]=set(directed[i])
    for i in np.flatnonzero(edge_rows):
        vm._hdmg_embedding_edges[i]=[j for j in directed[i] if i in sets[j]]+[j for j in directed[i] if i not in sets[j]]
    dirty_keys={vm._hdmg_base_keys[i] for i in dirty}
    semantic_keys=dirty_keys | {key for key,others in key_nearest.items() if set(others)&dirty_keys}
    semantic_rows=np.zeros(n,bool)
    for key in semantic_keys & by_key.keys():
        rows=by_key[key]
        changed=np.zeros(len(rows),bool)
        for part in config['semantic_parts'][key]:
            candidates=part['candidates'];nearest=part['nearest'];count=part['count']
            if count==0:
                continue
            affected_rows=dirty_mask[rows].copy()
            changed_candidates=candidates[dirty_mask[candidates]]
            if len(changed_candidates):
                affected_rows|=dirty_mask[nearest].any(axis=1)
                threshold=dist[rows[:,None],nearest].max(axis=1)
                affected_rows|=dist[np.ix_(rows,changed_candidates)].min(axis=1)<=threshold
            positions=np.flatnonzero(affected_rows)
            orders=np.argsort(dist[np.ix_(rows[positions],candidates)],axis=1,kind='stable')
            for position,order in zip(positions,orders):
                nearest[position]=candidates[order[candidates[order]!=rows[position]][:count]]
            changed|=affected_rows
        for position in np.flatnonzero(changed):
            i=rows[position]
            edge=[int(j) for part in config['semantic_parts'][key] for j in part['nearest'][position]]
            vm._hdmg_semantic_edges[i]=list(dict.fromkeys(edge))
        semantic_rows[rows[changed]]=True
    for key in dirty_keys:
        ids=by_key[key]
        vm._hdmg_key_entries[key]=int(ids[np.argmax(norm[ids]@unit(reps[ids].mean(axis=0)))])
    vm._hdmg_key_entry_ids=[vm._hdmg_key_entries[k] for k in vm._hdmg_key_entry_base_keys]
    vm._hdmg_key_entry_rep_matrix=vm._hdmg_rep_matrix_normed[vm._hdmg_key_entry_ids]
    vm._hdmg_global_entry=int(np.argmax(norm@unit(reps.mean(axis=0))))
    for i in dirty:
        vectors=vm._hdmg_nodes[i][1].vectors
        vm._hdmg_node_vec_counts[i]=len(vectors)
        vm._hdmg_node_vec_matrices[i]=(unit(np.stack(vectors)).astype(np.float32) if vectors else np.empty((0,reps.shape[1]),np.float32))
    vm._fast_hdmg_stats=dict(changed_centers=len(dirty),neighbor_rows_updated=len(changes),nodes=n,
        changed_edge_rows=np.flatnonzero(edge_rows|semantic_rows).tolist())
