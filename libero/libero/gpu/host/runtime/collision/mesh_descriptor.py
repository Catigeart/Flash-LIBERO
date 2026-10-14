"""Cold graph descriptor; pure NumPy, no CUDA/Warp imports."""
import numpy as np
def validate_mesh_static(static, *, edge_ends=None):
    vertex=np.asarray(static['mesh_vert'])
    if vertex.dtype!=np.float32 or vertex.ndim!=1 or vertex.size%3 or not np.isfinite(vertex).all():
        raise ValueError('mesh vertex dtype/shape/finite schema')
    names=('mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph')
    arrays={n:np.asarray(static[n]) for n in names}
    for n,a in arrays.items():
        if a.dtype!=np.int32 or a.ndim!=1:raise ValueError('mesh graph schema: '+n)
    adr,nums,gadr,graph=(arrays[n] for n in names)
    if not adr.shape==nums.shape==gadr.shape:raise ValueError('mesh metadata lengths')
    if np.any(adr<0) or np.any(nums<0) or np.any(3*(adr.astype(np.int64)+nums.astype(np.int64))>vertex.size):
        raise ValueError('mesh vertex address bounds')
    if edge_ends is not None and (edge_ends.dtype!=np.int32 or edge_ends.shape!=graph.shape):
        raise ValueError('mesh edge end descriptor schema')
    starts=sorted(set(int(a) for a in gadr if a>=0))
    for i,ga in enumerate(starts):
        end=starts[i+1] if i+1<len(starts) else graph.size
        if ga+2>end:raise ValueError('mesh graph header bounds')
        nv=int(graph[ga])
        if nv<=0 or ga+2+2*nv>end:raise ValueError('mesh graph local vertex bounds')
        owners=np.flatnonzero(gadr==ga)
        gv=graph[ga+2+nv:ga+2+2*nv]
        if np.any(gv<0) or any(np.any(gv>=nums[m]) for m in owners):
            raise ValueError('mesh graph global vertex bounds')
        edges=graph[ga+2+2*nv:end]
        for local,start in enumerate(graph[ga+2:ga+2+nv]):
            ei=int(start)
            if not 0<=ei<len(edges):raise ValueError('mesh edge address bounds')
            while ei<len(edges) and edges[ei]>=0:
                if edges[ei]>=nv:raise ValueError('mesh edge local vertex bounds')
                ei+=1
            if ei==len(edges) or edges[ei]!=-1:raise ValueError('mesh edge terminator bounds')
            if edge_ends is not None:edge_ends[ga+2+local]=ei
    return len(adr)
def build_mesh_descriptor(static):
    graph=np.asarray(static['mesh_graph'])
    ends=np.full(graph.shape,-1,dtype=np.int32)
    validate_mesh_static(static,edge_ends=ends)
    degree=0
    for ga in sorted(set(int(x) for x in np.asarray(static['mesh_graphadr']) if x>=0)):
        nv=int(graph[ga])
        degree=max(degree,int(np.max(ends[ga+2:ga+2+nv]-graph[ga+2:ga+2+nv])))
    no_graph=np.asarray(static['mesh_graphadr'])<0
    counts=np.asarray(static['mesh_vertnum'])
    scan=int(np.max(counts[no_graph])) if no_graph.any() else 0
    ends.flags.writeable=False
    return {'edge_ends':ends,'max_graph_degree':degree,'max_linear_vertices':scan,
            'all_meshes_have_graph':bool(len(counts)>0 and not no_graph.any()),
            'mesh_count':len(counts),'descriptor_bytes':int(ends.nbytes)}
