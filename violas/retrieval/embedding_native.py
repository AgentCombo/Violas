"""Native pure-embedding fast path over the existing VG/HDMG structures."""
import ctypes
from pathlib import Path
import numpy as np

class NativeEmbeddingHDMG:
    def __init__(self,corpus,pivots):
        self.c=corpus
        self.vectors=np.ascontiguousarray(corpus.vectors,np.float32)
        self.norms=np.sqrt(np.einsum('ij,ij->i',self.vectors,self.vectors,dtype=np.float64))
        centers=corpus.centers
        norm=np.linalg.norm(centers,axis=1,keepdims=True)
        self.centers=np.ascontiguousarray(centers/np.where(norm==0,1,norm),np.float32)
        self.pivots=np.ascontiguousarray(pivots,np.float32)
        self.edges=np.asarray([i for row in corpus.edges for i in row],np.int32)
        self.edge_offsets=np.asarray([0,*np.cumsum([len(row) for row in corpus.edges])],np.int32)
        self.members=np.concatenate(corpus.members).astype(np.int32)
        self.member_offsets=np.asarray([0,*np.cumsum([len(row) for row in corpus.members])],np.int32)
        self.entries=np.asarray(sorted(set(corpus.entries.values())),np.int32)
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('libembedding_hdmg_native.so').resolve()))
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.fn=self.lib.embedding_search
        self.fn.argtypes=[ctypes.c_int]*3+[f32,f64,f32,f32,i32,i32,i32,i32,i32]+[ctypes.c_int]*2+[f32]+[ctypes.c_int]*3+[i32,f64,i32]
        self.fn.restype=ctypes.c_int

    def search(self,query,group_budget=128,navigation_budget=256,exhaustive=False):
        query=np.ascontiguousarray(query,np.float32)
        out=np.empty(3,np.int32);scores=np.empty(3,np.float64);stats=np.empty(4,np.int32)
        count=self.fn(len(self.c.ids),len(self.c.members),self.vectors.shape[1],self.vectors,self.norms,
            self.centers,self.pivots,self.edges,self.edge_offsets,self.members,self.member_offsets,
            self.entries,len(self.entries),self.c.global_entry if self.c.global_entry is not None else -1,
            query,group_budget,navigation_budget,int(exhaustive),out,scores,stats)
        return [self.c.ids[i] for i in out[:count]],scores[:count].tolist(),dict(zip(
            ('scored_objects','searched_groups','nav_expansions','neighbor_reads'),map(int,stats)))
