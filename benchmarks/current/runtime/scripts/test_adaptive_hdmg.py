"""Independent small randomized checks against exhaustive set enumeration."""
from itertools import combinations
from types import SimpleNamespace
import numpy as np
from violas.retrieval import scoring as soft
from violas.retrieval.hdmg import AdaptiveHDMG

rng=np.random.default_rng(42)
for trial in range(40):
    vectors=rng.normal(size=(18,6)).astype(np.float32)
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    query=rng.normal(size=6).astype(np.float32)
    query/=np.linalg.norm(query)
    c=SimpleNamespace()
    c.ids=[f'id{i:02d}' for i in range(18)]
    c.vectors=vectors
    c.entity_ids=np.repeat([0,1],9)
    c.node_keys=np.repeat([0,1],3)
    c.members=[np.arange(i,i+3) for i in range(0,18,3)]
    c.centers=np.stack([vectors[m].mean(axis=0) for m in c.members])
    c.centers/=np.linalg.norm(c.centers,axis=1,keepdims=True)
    c.key_index={'A':0,'B':1}
    c.entries={'A':0,'B':3}
    c.global_entry=0
    # Deliberately disconnected graphs test metadata fallback safety.
    c.edges=[() for _ in c.members] if trial%2 else [(1,),(0,2),(1,),(4,),(3,5),(4,)]
    c.facets=[{'color':['red','blue','green'][int(rng.integers(3))],
               'flag':['yes','no'][int(rng.integers(2))]} for _ in c.ids]
    c.node_tokens=[set((k,v) for i in m for k,v in c.facets[i].items()) for m in c.members]
    c.token_nodes={t:np.array([i for i,tokens in enumerate(c.node_tokens) if t in tokens],dtype=int)
                   for tokens in c.node_tokens for t in tokens}
    c.objects={oid:{'description':{'entity':'A' if i<9 else 'B','facet_values':c.facets[i]}} for i,oid in enumerate(c.ids)}
    c.mixed_values=soft.Corpus.mixed_values
    c.semantic_scores=lambda plan:np.array([1.,.3])
    c.scores=lambda indices,q,p,semantic:{c.ids[i]:float(v) for i,v in zip(indices,
        c.mixed_values(np.einsum('ij,j->i',vectors[indices],q),c.entity_ids[indices],p,semantic))}
    plan={'required_entity':'A' if trial%3 else None,'entity_preference':'A',
          'hard_filters':[{'key':'flag','value':'yes'}] if trial%2 else [],
          'facet_dimension':'color' if trial%4 else None,'top_k':3,'lambda':.15,
          'category_preferences':{'red':{'lambda':.15,'quota':2},'blue':{'lambda':.05,'quota':1}},
          'embedding_weight':.65,'semantic_weight':.35}
    indices=[i for i in range(18) if (plan['required_entity'] is None or i<9) and
             all(c.facets[i][r['key']]==r['value'] for r in plan['hard_filters'])]
    scores=c.scores(indices,query,plan,c.semantic_scores(plan))
    def value(ids,sc):
        return soft.core.objective(ids,sc,c.objects,plan['facet_dimension'],plan['lambda'],plan['category_preferences'])
    optimum=max((value(ids,scores) for ids in combinations(scores,min(3,len(scores)))),default=0.)
    for batch in (False,True):
        ids,found,stats=AdaptiveHDMG(c).search(query,plan,budget=1,certify=True,batch_verification=batch)
        assert stats['certified'] and abs(value(ids,found)-optimum)<2e-6,(trial,optimum,value(ids,found))
print('PASS 40 randomized exhaustive checks: hard scopes, facets, quotas, disconnected graphs')
