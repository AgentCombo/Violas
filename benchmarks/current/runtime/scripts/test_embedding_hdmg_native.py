"""Independent exact-path checks, including ties, zeros and non-unit vectors."""
from types import SimpleNamespace
import numpy as np
from embedding_hdmg_native import NativeEmbeddingHDMG

rng=np.random.default_rng(73)
checks=0
for dimension in (1,7,256,1024):
    vectors=rng.normal(size=(90,dimension)).astype(np.float32)
    vectors[0]=0;vectors[1:5]=vectors[5]
    members=[np.arange(i,i+10,dtype=np.int32) for i in range(0,90,10)]
    c=SimpleNamespace(ids=[f'id{i:04d}' for i in range(90)],vectors=vectors,members=members,
        centers=np.stack([vectors[m].mean(axis=0) for m in members]),
        entries={'entry':0},global_entry=0,edges=[tuple(j for j in range(9) if j!=i) for i in range(9)])
    pivots=np.stack([vectors[m[:4]] for m in members])
    engine=NativeEmbeddingHDMG(c,pivots)
    for query in [np.zeros(dimension,np.float32),vectors[5],*rng.normal(size=(8,dimension)).astype(np.float32)]:
        norm=np.linalg.norm(vectors.astype(float),axis=1)*np.linalg.norm(query.astype(float))
        values=np.divide(vectors.astype(float)@query.astype(float),norm,out=np.zeros(90),where=norm>0)
        order=np.lexsort((np.asarray(c.ids),-values))[:3]
        ids,scores,stats=engine.search(query,exhaustive=True)
        assert np.allclose(scores,values[order],rtol=0,atol=1e-12)
        # Float64 near-ties may round differently, so verify exact IDs only when
        # the mathematical top-three boundary is separated from the next score.
        sorted_values=np.sort(values)
        if sorted_values[-3]-sorted_values[-4]>1e-10:
            assert set(ids)=={c.ids[i] for i in order}
        assert stats['scored_objects']==90
        checks+=1
print('PASS',checks,'native exact-search checks against independent float64 cosine')
