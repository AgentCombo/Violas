"""Vectorized deterministic equivalent of geometric_summary."""
import math
import numpy as np

def geometric_summary(ids,objects,representative_count=4):
    ordered=tuple(sorted(ids))
    matrix=np.stack([objects[oid]['normalized'] for oid in ordered]).astype(np.float32)
    center=matrix.mean(axis=0)
    norm=float(np.linalg.norm(center))
    if norm>0:
        center=(center/norm).astype(np.float32)
        cos=np.clip(matrix@center,-1,1)
        radius=float(np.arccos(cos).max())
        first=int(np.argmax(cos))
    else:
        center=np.zeros_like(center)
        radius=math.pi
        first=0
    chosen=[first]
    nearest=np.sum((matrix-matrix[first])**2,axis=1)
    while len(chosen)<min(representative_count,len(ordered)):
        work=nearest.copy()
        work[chosen]=-np.inf
        i=int(np.argmax(work))
        chosen.append(i)
        nearest=np.minimum(nearest,np.sum((matrix-matrix[i])**2,axis=1))
    return dict(unit_center=center,angular_radius=radius,representative_ids=tuple(ordered[i] for i in chosen))
