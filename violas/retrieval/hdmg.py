"""HDMG best-first expansion with exact hard-match metadata and safe continuation.

Uses only gallery vectors/metadata. No oracle or test-answer inputs.
Fixed budgets are approximate; certified mode visits further groups while a
centroid-radius bound plus a maximum facet reward can improve the selection.
"""
from collections import defaultdict, Counter
from heapq import nsmallest
import numpy as np
from . import scoring as soft


class AdaptiveHDMG:
    def select(self, pool, plan):
        if not plan['category_preferences'] and plan['lambda'] == 0:
            return nsmallest(max(0,plan['top_k']),pool,key=lambda oid:(-pool[oid],oid))
        return soft.select(pool,self.c,plan)

    def __init__(self, corpus):
        self.c = corpus
        self.node_of = np.full(len(corpus.ids), -1, np.int32)
        postings = defaultdict(list)
        self.radii = np.zeros(len(corpus.members), np.float64)
        self.pivots = []
        for node, members in enumerate(corpus.members):
            if np.any(self.node_of[members] >= 0):
                raise ValueError('Expected disjoint microclusters')
            self.node_of[members] = node
            vectors = corpus.vectors[members]
            center = corpus.centers[node].astype(np.float64)
            self.radii[node] = np.linalg.norm(vectors.astype(np.float64)-center, axis=1).max()
            # Four gallery-only representative members reduce mean-centroid bias.
            chosen = [int(np.argmax(vectors @ corpus.centers[node]))]
            nearest = vectors @ vectors[chosen[0]]
            for _ in range(1, min(4, len(members))):
                nearest[chosen] = np.inf
                chosen.append(int(np.argmin(nearest)))
                nearest = np.maximum(nearest, vectors @ vectors[chosen[-1]])
            self.pivots.append(vectors[(chosen * 4)[:4]])
        if np.any(self.node_of < 0):
            raise ValueError('Missing gallery objects')
        self.pivots = np.stack(self.pivots)
        for i, values in enumerate(corpus.facets):
            for token in values.items():
                postings[token].append(i)
        self.postings = {t:np.asarray(v, np.int32) for t,v in postings.items()}
        self.entities = {key:np.flatnonzero(corpus.entity_ids == i) for key,i in corpus.key_index.items()}
        self.empty = np.empty(0,np.int32)

    def search(self, query, plan, budget=8, certify=False, batch_verification=False, score_batch=1):
        c = self.c
        required = plan.get('required_entity')
        lists = [self.entities[required]] if required else [np.arange(len(c.ids))]
        lists.extend(self.postings.get((r['key'],r['value']),self.empty) for r in plan['hard_filters'])
        lists.sort(key=len)
        eligible = lists[0]
        for posting in lists[1:]:
            eligible = np.intersect1d(eligible, posting, assume_unique=True)
        if hasattr(c,'active'):
            eligible=eligible[c.active[eligible]]
        nodes = np.unique(self.node_of[eligible])
        if not len(nodes):
            return [],{},dict(unique_member_scores=0,searched_groups=0,eligible_count=0,certified=True)
        # If the budget covers every eligible group, navigation cannot prune
        # any work. Score their members in one kernel and run SELECT once.
        if len(nodes) <= budget:
            semantic = c.semantic_scores(plan)
            pool = c.scores(eligible,query,plan,semantic)
            selected = self.select(pool,plan)
            return selected,pool,dict(unique_member_scores=len(pool),searched_groups=len(nodes),
                eligible_count=len(eligible),eligible_groups=len(nodes),selected_groups=nodes.tolist(),
                center_scores=0,pivot_scores=0,neighbor_reads=0,graph_restarts=0,
                certified=True,route='all_eligible_groups')
        member_mask = np.zeros(len(c.ids),bool)
        member_mask[eligible] = True
        semantic = c.semantic_scores(plan)
        scores = np.full(len(c.members),-np.inf)
        cosines = np.einsum('ij,j->i',c.centers[nodes],query)
        pivot_cos = np.einsum('ipj,j->ip',self.pivots[nodes],query).max(axis=1)
        scores[nodes] = c.mixed_values(np.maximum(cosines,pivot_cos),c.node_keys[nodes],plan,semantic)
        # Bounds cover every member of every feasible group, including groups
        # not reachable via the filtered graph. Conservative rounding margin.
        if certify:
            upper = np.full(len(c.members),-np.inf)
            cap = np.minimum(1.000001, np.einsum('ij,j->i',c.centers[nodes].astype(np.float64),query.astype(np.float64))
                             + np.linalg.norm(query.astype(np.float64))*self.radii[nodes])
            upper[nodes] = c.mixed_values(cap,c.node_keys[nodes],plan,semantic)
        dimension = plan['facet_dimension']
        preferences = plan['category_preferences']
        max_bonus = np.zeros(len(c.members))
        # Node summaries may overestimate reward after hard filtering, which is
        # safe for pruning. Do not rescan all object metadata on every query.
        for value,pref in preferences.items():
            if pref['quota']>0:
                value_nodes=c.token_nodes.get((dimension,value),[])
                max_bonus[value_nodes]=np.maximum(max_bonus[value_nodes],pref['lambda'])
        navigation_bonus = max_bonus.copy()
        remaining = np.zeros(len(c.members),bool)
        remaining[nodes] = True
        frontier = np.zeros(len(c.members),bool)
        entry = int(c.global_entry) if plan['semantic_weight']==0 and required is None and c.global_entry is not None else int(c.entries[plan['entity_preference']])
        if remaining[entry]:
            frontier[entry] = True
        else:
            frontier[nodes[np.argmax(scores[nodes]+max_bonus[nodes])]] = True
        groups, pool, selected = [], {}, []
        # Keep only k best scored members per facet value for repeated SELECT.
        best = defaultdict(list)
        edge_reads, restarts, threshold = 0,0,-np.inf
        certified = False
        pending=[]
        while remaining.any():
            if len(groups)>=budget:
                if not certify:
                    break
                promising = remaining & (upper+max_bonus >= threshold-2e-6)
                if not promising.any():
                    certified = True
                    break
                remaining &= promising
                if batch_verification:
                    # A single batched distance kernel for all unprunable
                    # remaining groups. Honest stats retain every scored object.
                    tail=np.flatnonzero(remaining)
                    members=np.concatenate([c.members[n] for n in tail])
                    members=members[member_mask[members]]
                    pool.update(c.scores(members,query,plan,semantic))
                    groups.extend(tail.tolist())
                    remaining[:]=False
                    selected=soft.select(pool,c,plan)
                    certified=True
                    break
            choices = np.flatnonzero(frontier & remaining)
            if not len(choices):
                choices = np.flatnonzero(remaining)
                restarts += 1
            node = int(choices[np.argmax(scores[choices]+navigation_bonus[choices])])
            remaining[node] = False
            frontier[node] = False
            groups.append(node)
            neighbors = c.edges[node]
            edge_reads += len(neighbors)
            if neighbors:
                frontier[list(neighbors)] = True
            members = c.members[node][member_mask[c.members[node]]]
            pending.extend(members.tolist())
            defer_scores = not preferences and not certify
            if (defer_scores or len(groups)%score_batch) and remaining.any() and len(groups)<budget:
                continue
            members=np.asarray(pending,np.int32)
            pending=[]
            new = c.scores(members,query,plan,semantic)
            pool.update(new)
            # With no category preferences the selected prefix cannot affect
            # navigation. Select once at the end of fixed-budget search.
            if not preferences and not certify:
                continue
            touched = set()
            for oid,score in new.items():
                value = soft.core.label_of(c.objects,oid,dimension)
                best[value].append((oid,score))
                touched.add(value)
            for value in touched:
                best[value] = sorted(best[value],key=lambda v:(-v[1],v[0]))[:plan['top_k']]
            prefix = dict(v for values in best.values() for v in values)
            selected = soft.select(prefix,c,plan)
            counts = Counter(soft.core.label_of(c.objects,oid,dimension) for oid in selected)
            navigation_bonus.fill(0)
            for value,pref in preferences.items():
                if counts[value] < pref['quota']:
                    value_nodes = c.token_nodes.get((dimension,value),[])
                    navigation_bonus[value_nodes] = np.maximum(navigation_bonus[value_nodes],pref['lambda'])
            if certify and len(selected)==plan['top_k']:
                # Last greedy marginal = kth largest category-prefix marginal.
                last = selected[-1]
                value = soft.core.label_of(c.objects,last,dimension)
                weight,quota = soft.core.category_preference(preferences,value,dimension,plan['lambda'])
                threshold = pool[last]+(weight if counts[value]<=quota else 0)
        if not remaining.any():
            certified = True
        if not preferences and not certify:
            selected = self.select(pool,plan)
        return selected,pool,dict(unique_member_scores=len(pool),searched_groups=len(groups),
            eligible_count=len(eligible),eligible_groups=len(nodes),selected_groups=groups,
            center_scores=len(nodes),pivot_scores=4*len(nodes),neighbor_reads=edge_reads,
            graph_restarts=restarts,certified=certified,route='adaptive_hdmg')
