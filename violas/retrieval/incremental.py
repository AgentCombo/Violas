"""Native VG groups with batched HDMG updates and durable operation journaling."""
import copy
import os
import pickle
from collections import defaultdict
import numpy as np
from .store_base import VGStore
from . import directory
from .objects import make_object_map
from violas.storage.vectormap import VectorMap
from .hdmg_build import build_fast_hdmg
from .hdmg import AdaptiveHDMG
from .hdmg_update import update as hot_update, prepare as prepare_hot
from violas.storage.vectorgroup import VectorGroup
import violas.storage.vectormap as vectormap_module
from . import scoring as soft
from .geometry import geometric_summary


class IncrementalVGStore(VGStore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.wal=self.path.with_suffix('.wal')
        self.engine=None

    def rebuild(self):
        original=VectorMap.build_hdmg
        original_summary=directory._geometric_summary
        directory._geometric_summary=geometric_summary
        VectorMap.build_hdmg=build_fast_hdmg
        try:
            super().rebuild()
        finally:
            VectorMap.build_hdmg=original
            directory._geometric_summary=original_summary
        self.corpus=soft.Corpus(self.cache,self.directory)
        self.corpus.active=np.ones(len(self.corpus.ids),bool)
        self.engine=AdaptiveHDMG(self.corpus)
        self.prepare_mutations()
        self.node_lookup={(key,g.group_name):i for i,(key,g) in enumerate(self.cache['vectormap']._hdmg_nodes)}

    def insert(self,ids):
        if self.cache is None:
            self.initial_build(ids)
        else:
            self.mutate('insert',ids)

    def initial_build(self,ids):
        vm=VectorMap()
        vm.key_vectors=self.template['vectormap'].key_vectors.copy()
        by_entity=defaultdict(list)
        for oid in ids:
            obj=self.objects[oid]
            desc={'object_id':oid,'entity':obj['description']['entity'],'facet_values':dict(obj['description']['facet_values'])}
            # Same compact metadata projection and unit embedding as VDBs.
            vector=np.asarray(obj['normalized'],np.float32).copy()
            self.active[oid]={**obj,'vector':vector,'normalized':vector,'description':desc}
            by_entity[desc['entity']].append(oid)
        original_kmeans=vectormap_module.KMeans
        def one_restart(*args,**kwargs):
            kwargs['n_init']=1
            return original_kmeans(*args,**kwargs)
        vectormap_module.KMeans=one_restart
        try:
            for entity,oids in sorted(by_entity.items()):
                oids.sort()
                vectors=[self.active[oid]['vector'] for oid in oids]
                descriptions=[self.active[oid]['description'] for oid in oids]
                old=self.objects[oids[0]]['group']
                group=VectorGroup(entity,np.mean(vectors,axis=0),entity,vectors,descriptions,old.vector_type,old.group_type)
                vm.insert_with_auto_cluster(entity,group,{'entity':entity},alpha=.5)
        finally:
            vectormap_module.KMeans=original_kmeans
        locator={};entity_objects=defaultdict(list)
        for key,entry in vm.data.items():
            for group in entry['groups']:
                for pos,desc in enumerate(group.descriptions):
                    oid=desc['object_id']
                    locator[oid]=(key,group.group_name,pos)
                    self.routes[oid]=(key,group.group_name)
                    entity_objects[desc['entity']].append(oid)
        build_fast_hdmg(vm,embedding_k=16,semantic_intra_k=8)
        self.cache={'vectormap':vm,'object_locator':locator,'entity_objects':dict(entity_objects),'schemas':self.template['schemas']}
        original_summary=directory._geometric_summary
        directory._geometric_summary=geometric_summary
        try:
            self.directory=directory.build_semantic_directory(self.cache)
        finally:
            directory._geometric_summary=original_summary
        with self.path.open('wb') as f:
            pickle.dump({'cache':self.cache,'directory':self.directory},f,pickle.HIGHEST_PROTOCOL)
            f.flush();os.fsync(f.fileno())
        self.corpus=soft.Corpus(self.cache,self.directory)
        self.corpus.active=np.ones(len(self.corpus.ids),bool)
        self.engine=AdaptiveHDMG(self.corpus)
        self.prepare_mutations()
        self.node_lookup={(key,g.group_name):i for i,(key,g) in enumerate(vm._hdmg_nodes)}

    def prepare_mutations(self):
        c=self.corpus
        n,d=c.vectors.shape
        self.vector_buffer=np.empty((max(n+64,int(n*1.25)),d),np.float32)
        self.vector_buffer[:n]=c.vectors
        c.vectors=self.vector_buffer[:n]
        self.entity_nodes=defaultdict(list)
        for i,key in enumerate(self.cache['vectormap']._hdmg_base_keys):
            self.entity_nodes[key].append(i)
        self.posting_sets={token:set(values.tolist()) for token,values in self.engine.postings.items()}
        self.entity_sets={key:set(values.tolist()) for key,values in self.engine.entities.items()}
        prepare_hot(self.cache['vectormap'])

    def delete(self,ids):
        self.mutate('delete',ids)

    def update(self,ids):
        self.mutate('update',ids)

    def mutate(self,kind,ids,journal=True):
        vm=self.cache['vectormap']
        dirty=set()
        oldtokens=set()
        tokens_by_entity=defaultdict(set)
        changed_subsets=set()
        entities=set()
        removed_by_token=defaultdict(set)
        added_by_token=defaultdict(set)
        corpus=self.corpus
        new_ids=[oid for oid in ids if kind!='delete' and oid not in corpus.locator]
        if new_ids:
            first=len(corpus.ids)
            corpus.ids.extend(new_ids)
            corpus.locator.update({oid:first+i for i,oid in enumerate(new_ids)})
            corpus.facets.extend({} for _ in new_ids)
            size=len(corpus.ids)
            if size>len(self.vector_buffer):
                buffer=np.empty((max(size,int(len(self.vector_buffer)*1.5)),corpus.vectors.shape[1]),np.float32)
                buffer[:first]=corpus.vectors
                self.vector_buffer=buffer
            corpus.vectors=self.vector_buffer[:size]
            corpus.entity_ids=np.concatenate((corpus.entity_ids,np.full(len(new_ids),-1,int)))
            corpus.active=np.concatenate((corpus.active,np.zeros(len(new_ids),bool)))
            self.engine.node_of=np.concatenate((self.engine.node_of,np.full(len(new_ids),-1,np.int32)))
        for oid in ids:
            if kind in ('delete','update'):
                key,name,pos=self.cache['object_locator'][oid]
                node=self.node_lookup[(key,name)]
                group=vm._hdmg_nodes[node][1]
                # Positions may have shifted after another removal in this batch.
                pos=next(i for i,d in enumerate(group.descriptions) if d['object_id']==oid)
                group.vectors.pop(pos)
                old=group.descriptions.pop(pos)
                oldtokens.update(old.get('facet_values',{}).items())
                tokens_by_entity[old['entity']].update(old.get('facet_values',{}).items())
                changed_subsets.update((node,old['entity'],k,v) for k,v in old.get('facet_values',{}).items())
                entities.add(old['entity'])
                dirty.add(node)
                del self.active[oid]
                del self.cache['object_locator'][oid]
                slot=corpus.locator[oid]
                self.entity_sets[old['entity']].discard(slot)
                for token in old.get('facet_values',{}).items():removed_by_token[token].add(slot)
                corpus.active[slot]=False
                corpus.entity_ids[slot]=-1
                corpus.objects.pop(oid,None)
            if kind in ('insert','update'):
                obj=self.objects[oid]
                entity=obj['description']['entity']
                entities.add(entity)
                candidates=self.entity_nodes.get(entity,[])
                if not candidates:
                    raise ValueError('New entity insertion requires explicit group registration')
                query=np.asarray(obj['normalized']).reshape(-1)
                centers=vm._hdmg_rep_matrix_normed[candidates]
                node=candidates[int(np.argmax(centers@query))]
                key,group=vm._hdmg_nodes[node]
                group.vectors.append(np.asarray(obj['normalized']).copy())
                desc={'object_id':oid,'entity':obj['description']['entity'],'facet_values':dict(obj['description']['facet_values'])}
                group.descriptions.append(desc)
                self.active[oid]={**obj,'vector':group.vectors[-1],'description':desc,'normalized':query.copy()}
                self.routes[oid]=(key,group.group_name)
                dirty.add(node)
                oldtokens.update(desc.get('facet_values',{}).items())
                tokens_by_entity[desc['entity']].update(desc.get('facet_values',{}).items())
                changed_subsets.update((node,desc['entity'],k,v) for k,v in desc.get('facet_values',{}).items())
                slot=corpus.locator[oid]
                corpus.vectors[slot]=query
                corpus.entity_ids[slot]=corpus.key_index[entity]
                corpus.active[slot]=True
                corpus.facets[slot]=desc.get('facet_values',{})
                self.engine.node_of[slot]=node
                self.entity_sets.setdefault(entity,set()).add(slot)
                for token in desc.get('facet_values',{}).items():added_by_token[token].add(slot)
                corpus.objects[oid]=self.active[oid]
        # Empty groups are retained as tombstones for stable graph/node IDs.
        # Their centroids remain routing waypoints; they cannot return members.
        subset_keys=defaultdict(list)
        previous_summaries={}
        for summary_key in changed_subsets:
            summary=self.directory['subset_summaries'].get(summary_key)
            if summary is not None:
                subset_keys[summary_key[0]].append(summary_key)
                previous_summaries[summary_key]=summary
        dirty_by_entity=defaultdict(set)
        member_buckets={}
        for node in dirty:
            dirty_by_entity[corpus.entities[corpus.node_keys[node]]].add(node)
        for node in dirty:
            key,group=vm._hdmg_nodes[node]
            if group.vectors and kind!='delete':
                group.representative=np.mean(group.vectors,axis=0)
            members=[]
            for pos,desc in enumerate(group.descriptions):
                oid=desc['object_id']
                self.cache['object_locator'][oid]=(key,group.group_name,pos)
                members.append(corpus.locator[oid])
            corpus.members[node]=np.asarray(members,np.int32)
            tokens=set((k,v) for i in members for k,v in corpus.facets[i].items())
            oldtokens.update(corpus.node_tokens[node])
            oldtokens.update(tokens)
            entity=corpus.entities[corpus.node_keys[node]]
            tokens_by_entity[entity].update(corpus.node_tokens[node])
            tokens_by_entity[entity].update(tokens)
            corpus.node_tokens[node]=tokens
            self.directory['node_members'][node]=tuple(corpus.ids[i] for i in members)
            if members and kind!='delete':
                vectors=corpus.vectors[members]
                center=vectors.mean(axis=0)
                corpus.centers[node]=center
                self.directory['centers'][node]=center
                self.engine.radii[node]=np.linalg.norm(vectors.astype(np.float64)-center.astype(np.float64),axis=1).max()
                chosen=[int(np.argmax(vectors@center))]
                near=vectors@vectors[chosen[0]]
                for _ in range(1,min(4,len(members))):
                    near[chosen]=np.inf
                    chosen.append(int(np.argmin(near)))
                    near=np.maximum(near,vectors@vectors[chosen[-1]])
                self.engine.pivots[node]=vectors[(chosen*4)[:4]]
                self.directory['radii'][node]=self.engine.radii[node]
                self.directory['node_summaries'][node]=geometric_summary(
                    self.directory['node_members'][node],self.active)
            for key in subset_keys[node]:
                del self.directory['subset_summaries'][key]
            buckets=defaultdict(list)
            for i in members:
                oid=corpus.ids[i]
                desc=self.active[oid]['description']
                for dimension,value in desc.get('facet_values',{}).items():
                    buckets[(desc['entity'],dimension,value)].append(oid)
            for (entity,dimension,value),oids in buckets.items():
                summary_key=(node,entity,dimension,value)
                member_buckets[summary_key]=tuple(sorted(oids))
                if summary_key not in changed_subsets:continue
                if kind=='delete':
                    summary=dict(previous_summaries[summary_key])
                    summary['representative_ids']=tuple(sorted(oids)[:4])
                else:
                    summary=geometric_summary(oids,self.active)
                self.directory['subset_summaries'][summary_key]=summary
        vm._fast_member_dirty=dirty
        if self.graph and kind!='delete':
            hot_update(vm,dirty)
        elif self.graph:
            # Logical deletion preserves valid conservative geometric caps and
            # routing edges, like tombstone deletion; physical recentering is not
            # required for visibility and is a separate maintenance operation.
            for node in dirty:
                members=corpus.members[node]
                vm._hdmg_node_vec_matrices[node]=corpus.vectors[members].copy()
                vm._hdmg_node_vec_counts[node]=len(members)
                summary=dict(self.directory['node_summaries'][node])
                summary['representative_ids']=tuple(self.directory['node_members'][node][:4])
                self.directory['node_summaries'][node]=summary
                if len(members):
                    self.engine.pivots[node]=corpus.vectors[(members.tolist()*4)[:4]]
            vm._fast_hdmg_stats={'changed_edge_rows':[]}
        node_adds=defaultdict(list)
        for node in dirty:
            for token in corpus.node_tokens[node]:node_adds[token].append(node)
        dirty_mask=np.zeros(len(corpus.members),bool)
        dirty_mask[list(dirty)]=True
        for token in oldtokens:
            prior_nodes=corpus.token_nodes.get(token,np.empty(0,np.int32))
            kept=prior_nodes[~dirty_mask[prior_nodes]]
            corpus.token_nodes[token]=np.sort(np.concatenate((kept,np.asarray(node_adds[token],np.int32))))
        for token in removed_by_token.keys() | added_by_token.keys():
            values=self.posting_sets.setdefault(token,set())
            values.difference_update(removed_by_token[token])
            values.update(added_by_token[token])
            self.engine.postings[token]=np.asarray(sorted(values),np.int32)
        for entity in entities:
            indices=np.asarray(sorted(self.entity_sets[entity]),np.int32)
            self.engine.entities[entity]=indices
            self.cache['entity_objects'][entity]=[corpus.ids[i] for i in indices]
            postings=self.directory['postings'].setdefault(entity,{})
            for dimension,value in tokens_by_entity[entity]:
                values=postings.setdefault(dimension,{})
                kept=[r for r in values.get(value,[]) if r['node_id'] not in dirty]
                for node in dirty_by_entity[entity]:
                    summary=self.directory['subset_summaries'].get((node,entity,dimension,value))
                    if summary is None:continue
                    key,group=vm._hdmg_nodes[node]
                    oids=member_buckets.get((node,entity,dimension,value),())
                    kept.append(dict(node_id=node,physical_key=key,group_name=group.group_name,
                        count=len(oids),object_ids=oids,summary=summary))
                values[value]=sorted(kept,key=lambda r:(-r['count'],r['node_id']))
        self.directory['object_count']=len(self.active)
        for i in vm._fast_hdmg_stats.get('changed_edge_rows',range(len(corpus.edges))):
            corpus.edges[i]=tuple(sorted(set(vm._hdmg_embedding_edges[i])|set(vm._hdmg_semantic_edges[i])))
        corpus.entries=dict(vm._hdmg_key_entries)
        corpus.global_entry=vm._hdmg_global_entry
        if journal:
            matrix=np.stack([self.active[oid]['normalized'] for oid in ids]) if kind!='delete' else None
            metadata=[self.active[oid]['description'] for oid in ids] if kind!='delete' else []
            with self.wal.open('ab') as f:
                pickle.dump(('vg_wal_v2',kind,list(ids),matrix,metadata),f,protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

    def query(self,query,plan):
        return self.engine.search(query,plan,certify=True,batch_verification=True)

    def verify(self,expected,touched,deleted):
        super().verify(expected,touched,deleted)
        assert {oid for oid,i in self.corpus.locator.items() if self.corpus.active[i]}==expected
        for oid in touched:
            if not deleted:
                assert np.allclose(self.corpus.vectors[self.corpus.locator[oid]],self.objects[oid]['normalized'])
                assert self.corpus.objects[oid]['description']['facet_values']==self.objects[oid]['description']['facet_values']

    def recover(self):
        with self.path.open('rb') as f:
            saved=pickle.load(f)
        self.cache,self.directory=saved['cache'],saved['directory']
        self.active=make_object_map(self.cache)
        self.routes={oid:(key,name) for oid,(key,name,pos) in self.cache['object_locator'].items()}
        self.corpus=soft.Corpus(self.cache,self.directory)
        self.corpus.active=np.ones(len(self.corpus.ids),bool)
        self.engine=AdaptiveHDMG(self.corpus)
        self.prepare_mutations()
        self.node_lookup={(key,g.group_name):i for i,(key,g) in enumerate(self.cache['vectormap']._hdmg_nodes)}
        if self.wal.exists():
            with self.wal.open('rb') as f:
                while True:
                    try:
                        record=pickle.load(f)
                    except EOFError:
                        break
                    if record[0]=='vg_wal_v2':
                        _,kind,ids,matrix,metadata=record
                        records={oid:dict(vector=matrix[i],normalized=matrix[i],description=metadata[i]) for i,oid in enumerate(ids)} if kind!='delete' else {}
                    else:
                        kind,ids,records=record
                    for oid,obj in records.items():
                        self.objects[oid]={**self.objects[oid],**obj}
                    self.mutate(kind,ids,journal=False)
