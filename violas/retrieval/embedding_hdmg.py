"""Embedding-only HDMG navigation, with batched graph and member scoring."""
import heapq
import numpy as np


class EmbeddingHDMG:
    def __init__(self,corpus,pivots):
        self.c=corpus
        self.pivots=pivots
        centers=corpus.centers
        norms=np.linalg.norm(centers,axis=1,keepdims=True)
        self.centers=centers/np.where(norms==0,1,norms)
        self.entries=np.asarray(sorted(set(corpus.entries.values())),np.int32)

    def search(self,query,group_budget=24,navigation_budget=64):
        c=self.c
        priorities={}
        frontier=[]
        seen=np.zeros(len(c.members),bool)
        def discover(nodes):
            nodes=np.asarray(nodes,np.int32)
            nodes=nodes[~seen[nodes]]
            if not len(nodes):return
            seen[nodes]=True
            centers=np.einsum('ij,j->i',self.centers[nodes],query)
            pivot=np.einsum('ipj,j->ip',self.pivots[nodes],query).max(axis=1)
            scores=np.maximum(centers,pivot)
            for node,score in zip(nodes,scores):
                node=int(node);score=float(score)
                priorities[node]=score
                heapq.heappush(frontier,(-score,node))
        entry_scores=np.einsum('ij,j->i',self.centers[self.entries],query)
        seeds=self.entries[np.argsort(-entry_scores,kind='stable')[:3]].tolist()
        if c.global_entry is not None:seeds.append(c.global_entry)
        discover(sorted(set(seeds)))
        expanded=0
        edges=0
        while frontier and expanded<navigation_budget:
            _,node=heapq.heappop(frontier)
            expanded+=1
            edges+=len(c.edges[node])
            discover(c.edges[node])
        groups=heapq.nsmallest(group_budget,priorities,key=lambda n:(-priorities[n],n))
        members=np.concatenate([c.members[n] for n in groups])
        values=(1+np.einsum('ij,j->i',c.vectors[members],query))/2
        positions=np.lexsort((members,-values))[:3]
        ids=[c.ids[members[i]] for i in positions]
        scores={c.ids[members[i]]:float(values[i]) for i in positions}
        return ids,scores,dict(unique_member_scores=len(members),searched_groups=len(groups),
            nav_expansions=expanded,center_scores=len(priorities)+len(self.entries),
            pivot_scores=4*len(priorities),neighbor_reads=edges,selected_groups=groups)
