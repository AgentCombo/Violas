"""Check graph topology against the original algorithm, including moving centers."""
import copy
import numpy as np
from violas.storage.vectormap import VectorMap
from violas.storage.vectorgroup import VectorGroup
from violas.retrieval.hdmg_build import build_fast_hdmg

rng=np.random.default_rng(73)
vm=VectorMap()
for e in range(4):
    entity=f'e{e}'
    vm.key_vectors[entity]=rng.normal(size=12).astype(np.float32)
    for j in range(6):
        v=rng.normal(size=(4,12)).astype(np.float32)
        desc=[{'object_id':f'{entity}/{j}/{i}','entity':entity,'facet_values':{'color':'red' if i%2 else 'blue'}} for i in range(4)]
        group=VectorGroup(f'{entity}-g{j}',v.mean(axis=0),'',list(v),desc)
        vm.insert(f'{entity}-{j:04d}',group,{'entity':entity})
        # The original base-key convention strips trailing -NNNN.
for iteration in range(3):
    slow=copy.deepcopy(vm)
    slow.build_hdmg(embedding_k=6,semantic_intra_k=3)
    build_fast_hdmg(vm,embedding_k=6,semantic_intra_k=3)
    assert slow._hdmg_embedding_edges==vm._hdmg_embedding_edges
    assert slow._hdmg_semantic_edges==vm._hdmg_semantic_edges
    assert slow._hdmg_key_entries==vm._hdmg_key_entries
    assert slow._hdmg_global_entry==vm._hdmg_global_entry
    for node in (1,8):
        group=vm._hdmg_nodes[node][1]
        group.vectors[0]=rng.normal(size=12).astype(np.float32)
        group.representative=np.mean(group.vectors,axis=0)
        vm._fast_member_dirty={1,8}
print('PASS fast HDMG topology and changed-center maintenance match original')
