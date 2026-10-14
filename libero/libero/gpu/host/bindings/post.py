"""Handwritten CUDA post-RNE, site force/torque sensors and strict N publication.

No arithmetic generation: the Python stage list supplies only fixed raw ABI
bindings. Ordered tree/contact work runs on lane zero of one block per world.
This is the existing elliptic, no-equality two force/torque sensor family.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
import numpy as np
import warp as wp
from libero.gpu.host.build.native import NativePlan


ROOT38_STATIC_SHA={'parent': 'b9bdcd24087e82e96e111007c07148013ccc73d21769d71b51ca569ea3a6dd5b', 'dofadr': 'c5b6fb754c2d61ce6f28af20560999e93c8c6ba9cd56e32448254af9cfa84632', 'dofnum': '4e95c2b048cd556411d3a1ec0096b967d3c8bc6147014428c50fab1e5ad4f86e', 'root': '85ada57e1f601e962d705f389285adb4e74f450bc00672240dfef7399d82457f', 'geom_body': 'e8d2f16977398facf3b59964fbbc8335fdbbf6cea0aff1395da828760c00385b', 'sensor.sensor_type': '04cf609765e0871139d5b23f097afce5bd703e1dd01607e6a94b681d9041668f', 'sensor.sensor_site': '01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d', 'sensor.sensor_adr': '9ae4c9d0864c30005acac91f21015f39e35691a056d606a097c8cf8380275a17', 'sensor.site_body': '1ac18868a7b9c7067db873909bd18784393291deca979bd4ffb26feec880eb2d', 'sensor.body_root': '85ada57e1f601e962d705f389285adb4e74f450bc00672240dfef7399d82457f'}


def _installer_index_profile(profile):
    # The measured real-input descriptor is a standalone held-input source
    # API. This mutable long-lived adapter has no bias lifetime authority.
    return None if profile == 'post_real_index_cache0_20261013_r05' else profile


def _reject_combined_profiles(index_profile, function_profile, root_profile):
    if root_profile is not None and (index_profile is not None or function_profile is not None):
        raise ValueError('Post root and index/function qualifications need separate standalone plans')


def _qualification_bias_enabled(cache_args, adapter):
    # Mirror the original per-world device predicate at explicit cold opt-in.
    # Merely having bound cache arrays does not imply bias=True. Device checks
    # still execute each call; this descriptor grants no mutable-owner reuse.
    worlds = adapter.capacity
    if type(worlds) is not int or worlds not in (1, 2):
        return None
    if not cache_args:
        return False
    if len(cache_args) not in (6, 7):
        return None
    if len(cache_args) == 7 and cache_args[-1] != 0:
        return False
    rows = [np.asarray(value.numpy()).reshape(-1) for value in
        (cache_args[3], cache_args[2], cache_args[4], cache_args[5],
         adapter.state_epoch, adapter.forward_epoch)]
    if any(row.shape != (worlds,) for row in rows):
        return None
    if any(row.dtype != np.dtype('int32') for row in rows[:2]) or any(
            row.dtype != np.dtype('int64') for row in rows[2:]):
        return None
    valid, status, state, forward, expected_state, expected_forward = rows
    enabled = (valid == 1) & (status == 0) & (state == expected_state) & (forward == expected_forward)
    if bool(np.all(enabled)):
        return True
    if not bool(np.any(enabled)):
        return False
    return None


def install(bundle, root, *, dataflow_profile="compatible",
            contact_index_mode="auto", radix_min_contacts=256,
            contact_topology_cache=True, contact_index_qualification=None,
            post_function_qualification=None, root_clear_qualification=None):
    if dataflow_profile not in ("compatible", "fast"):
        raise ValueError("unknown post dataflow profile")
    contact_modes = {"auto": 0, "scan": 1, "radix": 2}
    if contact_index_mode not in contact_modes:
        raise ValueError("post contact index mode must be auto, scan or radix")
    if isinstance(radix_min_contacts, bool) or not isinstance(radix_min_contacts, int) or radix_min_contacts < 0:
        raise ValueError("post radix contact threshold must be a nonnegative integer")
    if not isinstance(contact_topology_cache, bool):
        raise ValueError("post contact topology cache requires a boolean")
    adapter = bundle.post
    if adapter is None:
        return {'installed': False, 'reason': 'no post adapter'}
    adapter.validate()
    p, s = adapter.post, adapter.post.sensor
    if hasattr(p, 'validate_bindings'):
        p.validate_bindings()
    dimensions = (getattr(p, 'nbody', 38), getattr(p, 'nv', 43),
                  getattr(p, 'nsite', 31), getattr(p, 'ngeom', 270))
    b, v, ns, ng = map(int, dimensions)
    if post_function_qualification is not None and contact_index_qualification is not None:
        raise ValueError('Post function and index qualifications need separate plans')
    _reject_combined_profiles(contact_index_qualification, post_function_qualification, root_clear_qualification)
    installer_index_profile = _installer_index_profile(contact_index_qualification)
    from libero.gpu.host.bindings.post_qualification import select_post_index
    index_qualification = select_post_index(installer_index_profile,
        (b,v,ns,ng),p.contact_capacity,p.efc_capacity,
        contact_index_mode=contact_index_mode,radix_min_contacts=radix_min_contacts,
        worlds=adapter.capacity,contact_topology_cache=contact_topology_cache,
        dataflow_profile=dataflow_profile,velocity_bias_cache=False)
    if contact_index_qualification == 'post_real_index_cache0_20261013_r05':
        index_qualification = {**index_qualification,
            'profile': contact_index_qualification,
            'reason': 'Real-input source qualification is standalone held-input only; this long-lived Post installer keeps literal original.',
            'persistent_bias_qualification': False,
            'real_family_production_selector_qualified': False}
    if root_clear_qualification not in (None, 'post_root38_fast_20261012_r01'):
        raise ValueError('unknown isolated root-clear qualification')
    root_selected = (root_clear_qualification is not None and dataflow_profile == 'fast'
        and (b,v,ns,ng,p.contact_capacity,p.efc_capacity) == (38,43,2,114,529,3174)
        and adapter.capacity == 1 and contact_index_mode == 'radix'
        and radix_min_contacts == 256 and contact_topology_cache is True)
    if root_selected:
        expected = ROOT38_STATIC_SHA
        for name, digest in expected.items():
            value = getattr(s, name[7:]) if name.startswith('sensor.') else getattr(p, name)
            host = value.numpy()
            root_selected = root_selected and host.dtype == np.dtype('int32') and __import__('hashlib').sha256(host.tobytes()).hexdigest() == digest
    root_qualification = dict(selected=root_selected, profile=root_clear_qualification,
        source_route='post_function_root38.cuh' if root_selected else 'post_dataflow_original.cuh',
        defines={}, reason='Exact cold b38/W1/fast/cacheenabled/static-table fixture; NC512 device guard, otherwise literal original',
        static_shared_upper_bound_bytes=32*4+9*32*8+10*4,
        performance_scope='Controlled full-capacity local standalone Post only; no factory/joint or task traffic claim')
    # Shared validation reuse is legal only for the same immutable input views.
    for sensor_name, post_name in (('cfrc','cfrc_int'),('com','subtree_com'),
            ('site_pos','site_xpos'),('site_mat','site_xmat'),('active','active')):
        if getattr(s,sensor_name).ptr != getattr(p,post_name).ptr:
            raise ValueError('post shared pipeline requires sensor/post alias: '+sensor_name)
    if s.result.shape[1] != 6 or p.cfrc_ext.shape[1] != 6*b:
        raise ValueError('post shared pipeline layout mismatch')
    gate = [getattr(adapter, k) for k in ('active', 'first_error', 'd_valid',
        'n_valid', 'n_error', 'n_written', 'n_attempt', 'expected_consumed',
        'state_epoch', 'state_consumed', 'forward_epoch', 'forward_consumed')]+[p.active]
    names = ('active','qvel','qacc','cinert','cvel','cdof','cdof_dot','xipos',
        'subtree_com','site_xpos','site_xmat','xfrc_applied','efc_force','ncon',
        'nefc','ne','contact_dim','contact_efc_address','contact_geom1',
        'contact_geom2','contact_frame','contact_pos','parent','dofadr','dofnum',
        'root','geom_body','gravity')
    rne = [getattr(p,k) for k in names]+[p.contact_capacity,p.efc_capacity,
        p.cacc,p.cfrc_ext,p.cfrc_int,p.contact_force,p.status,b,v,ns,ng]
    sensors = [getattr(s,k) for k in ('cfrc','com','site_pos','site_mat','active',
        'sensor_type','sensor_site','sensor_adr','site_body','body_root','result','status')]+[b,ns]
    commit_args = [p.active,p.status,s.status,adapter.first_error,
        adapter.expected_consumed,p.cfrc_ext,adapter._int_view,s.result,
        adapter.public['cfrc_ext'],adapter.public['cfrc_int'],adapter.public['sensor'],
        adapter.valid,adapter.written,p.cfrc_ext.shape[1],s.result.shape[1]]
    cache = getattr(getattr(bundle, 'dynamics', None), 'cuda_velocity_bias_cache', None)
    cache_args = []
    dynamics = getattr(bundle, 'dynamics', None)
    rne_model = getattr(dynamics, 'rne_model', None)
    cache_gravity_matches = rne_model is None or np.array_equal(rne_model.gravity.numpy(),p.gravity.numpy())
    cache_layout_matches = cache is not None and getattr(cache, 'layout', None) == ('component_body', 1, b)
    if dataflow_profile == 'fast' and cache is not None and cache_gravity_matches and cache_layout_matches:
        if cache.acc.shape != (adapter.capacity,6*b) or cache.body_force.shape != (adapter.capacity,6*b):
            raise ValueError('post velocity cache body layout mismatch')
        cache_args = [cache.acc, cache.body_force, cache.status, cache.valid,
                      cache.state_written, cache.forward_written]
        if rne_model is not None:
            gravity_disabled = rne_model.gravity_disabled
            if hasattr(gravity_disabled, 'numpy'):
                gravity_disabled = np.asarray(gravity_disabled.numpy()).reshape(-1)[0]
            cache_args.append(int(gravity_disabled))
    # Resolve the measured independent cold profile after preserving the
    # original cache owner/layout/gravity validation. No GPU math is changed.
    qualification_bias_enabled = False
    if installer_index_profile is not None or post_function_qualification is not None:
        qualification_bias_enabled = _qualification_bias_enabled(cache_args, adapter)
    if installer_index_profile is not None:
        index_qualification = select_post_index(installer_index_profile,
            (b,v,ns,ng),p.contact_capacity,p.efc_capacity,
            contact_index_mode=contact_index_mode,radix_min_contacts=radix_min_contacts,
            worlds=adapter.capacity,contact_topology_cache=contact_topology_cache,
            dataflow_profile=dataflow_profile,velocity_bias_cache=qualification_bias_enabled)
    # Cold body topology proves levels and descending sibling summation order.
    from libero.gpu.model.topology import prepare_body_topology
    topology = prepare_body_topology(p.parent.numpy(), adapter.device)
    if topology.proof['nbody'] != b:
        raise ValueError('post body topology dimensions')
    from libero.gpu.host.bindings.post_function_qualification import select_post_function
    function_qualification = select_post_function(post_function_qualification,
        (b,v,ns,ng),p.contact_capacity,p.efc_capacity,worlds=adapter.capacity,
        contact_index_mode=contact_index_mode,radix_min_contacts=radix_min_contacts,
        parent=p.parent.numpy(),contact_topology_cache=contact_topology_cache,
        dataflow_profile=dataflow_profile,velocity_bias_cache=qualification_bias_enabled)
    source_qualification = function_qualification if post_function_qualification is not None else index_qualification
    if root_clear_qualification is not None: source_qualification = root_qualification
    if p.contact_capacity > (2147483647 // 2):
        raise ValueError('post contact event index exceeds int32')
    workspace = {
        'body_index': wp.zeros((adapter.capacity, 2 * p.contact_capacity), dtype=wp.int32, device=adapter.device),
        'body_offsets': wp.zeros((adapter.capacity, b + 1), dtype=wp.int32, device=adapter.device),
        'body_counts': wp.zeros((adapter.capacity, b), dtype=wp.int32, device=adapter.device),
        'event_ping': wp.empty((adapter.capacity, 2 * p.contact_capacity), dtype=wp.int32, device=adapter.device),
        'event_pong': wp.empty((adapter.capacity, 2 * p.contact_capacity), dtype=wp.int32, device=adapter.device),
        'cached_event_body': wp.empty((adapter.capacity, 2 * p.contact_capacity), dtype=wp.int32, device=adapter.device),
        'cached_contact_meta': wp.zeros((adapter.capacity, 2), dtype=wp.int32, device=adapter.device),
    }
    topology_args = [workspace[k] for k in ('body_index', 'body_offsets', 'body_counts')]
    topology_args += [topology.arrays['level_offsets'], topology.arrays['level_bodies'],
        int(topology.proof['nlevels']), topology.arrays['children_offsets'], topology.arrays['children_indices']]
    topology_args += [workspace['event_ping'], workspace['event_pong'],
                      contact_modes[contact_index_mode], radix_min_contacts]
    topology_args += [workspace['cached_event_body'], workspace['cached_contact_meta'], int(contact_topology_cache)]
    if function_qualification['selected'] and function_qualification['chains']:
        from libero.gpu.host.bindings.post_topology import post_chains
        chains = post_chains(p.parent.numpy())
        for name in ('chain_offsets','chain_bodies','chain_level_offsets','chain_level_indices'):
            workspace[name] = wp.array(chains[name], dtype=wp.int32, device=adapter.device)
            topology_args.append(workspace[name])
        topology_args.append(int(chains['nchain_levels']))
    adapter.cuda_post_contact_index_config = (contact_modes[contact_index_mode], radix_min_contacts, int(contact_topology_cache))
    cache_enabled = bool(cache_args)
    # NativePlan binds real typed arrays. A disabled cache uses harmless owners,
    # with gravity_disabled=1 preventing any numerical access to them.
    if not cache_args:
        disabled = wp.zeros(adapter.capacity, dtype=wp.int32, device=adapter.device)
        disabled_epoch = wp.zeros(adapter.capacity, dtype=wp.int64, device=adapter.device)
        cache_args = [p.cacc, p.cfrc_ext, disabled, disabled, disabled_epoch, disabled_epoch, 1]
    elif len(cache_args) == 6:
        cache_args.append(0)
    stages = [('libero_post::run', gate+ rne+ sensors+ commit_args+
        [int(dataflow_profile == 'compatible')]+cache_args+topology_args, 'block')]
    plan = NativePlan('post_rne_sensor_dataflow',
        [(GPU_ROOT / 'cuda/postprocess/post.cuh'), (GPU_ROOT / 'cuda/postprocess' / source_qualification['source_route'])],
        stages,adapter.capacity,Path(root)/'post',device=adapter.device,
        dynamic_shared_bytes=(18*b+6)*8, stage_barriers=False,
        defines=source_qualification['defines'])
    # Shared cache arrays changed physical ABI without changing their extents.
    # Seal layout and the exact producer owners for both standalone and joint N.
    cache_owners = tuple(getattr(cache, k) for k in ('acc', 'body_force', 'status',
        'valid', 'state_written', 'forward_written')) if cache_enabled else ()
    contact_owners = tuple(workspace.items())
    def cache_prevalidate():
        if getattr(adapter, 'cuda_post_contact_workspace', None) is not workspace or any(
                workspace.get(k) is not owner for k, owner in contact_owners):
            raise ValueError('post contact workspace owner drift; rebuild post and joint N tail')
        if not cache_enabled:
            return
        if getattr(bundle.dynamics, 'cuda_velocity_bias_cache', None) is not cache:
            raise ValueError('post velocity cache owner drift')
        if getattr(cache, 'layout', None) != ('component_body', 1, b):
            raise ValueError('post velocity cache physical layout drift')
        if any(getattr(cache, k) is not owner for k, owner in zip(
                ('acc', 'body_force', 'status', 'valid', 'state_written', 'forward_written'), cache_owners)):
            raise ValueError('post velocity cache array owner drift')
    adapter.cuda_post_cache_prevalidate = cache_prevalidate
    original = adapter.enqueue
    def enqueue(self, stream=None):
        self.validate(stream)
        if hasattr(self.post,'validate_bindings'):
            self.post.validate_bindings()
        cache_prevalidate()
        plan.launch(self.stream)
    adapter.enqueue = MethodType(enqueue,adapter)
    adapter.cuda_post_plan = plan
    adapter.cuda_post_contact_workspace = workspace
    def invalidate_contact_cache(stream=None):
        """Call after a QA restore overwrites CSR without restoring its key/meta.

        Normal recovery changes input keys and needs no host invalidation. This
        private workspace has one ordered producer, shared by post and N tail.
        """
        with wp.ScopedStream(stream or adapter.stream):
            workspace['cached_contact_meta'].zero_()
    adapter.cuda_post_contact_cache_invalidate = invalidate_contact_cache
    adapter.cuda_post_body_topology = topology
    adapter.cuda_post_index_qualification = index_qualification
    adapter.cuda_post_function_qualification = function_qualification
    adapter.cuda_post_root_clear_qualification = root_qualification
    adapter.cuda_post_original = original
    # Both prepared fixed and generic forward callbacks look up post.enqueue at
    # invocation time. Set the callback explicitly to document the active path.
    forward = bundle.runtime.forward
    forward.acceleration_sensor = lambda mask, stream: adapter.enqueue(stream=stream)
    return {'installed': True,'manifest': plan.manifest,'dimensions': dimensions,
        'stages': len(stages),'kernels': 1,'dataflow_profile': dataflow_profile,
        'shared_bytes': (18*b+6)*8,
        'velocity_bias_cache_layout': ('component_body', 1, b) if cache_enabled else None,
        'static_shared_bytes': source_qualification.get('static_shared_upper_bound_bytes',32*4 + 9*32*8 + 10*4),
        'static_shared_is_upper_bound': bool(source_qualification['selected']),
        'contact_index_qualification': index_qualification,
        'post_function_qualification': function_qualification,
        'root_clear_qualification': root_qualification,
        'contact_decode_tile': function_qualification['contact_tile'],
        'spatial_workspace_layout': 'component planes; public body-major','velocity_bias_cache': cache_enabled,'private_qacc': 'explicit rne qa pointer, world-strided',
        'body_topology_sha256': topology.source_sha256,
        'body_contact_workspace_bytes': adapter.capacity * (8*p.contact_capacity+2*b+3)*4,
        'contact_event_order': '2*contact+side; each body consumes ascending original event indices',
        'contact_index_mode': contact_index_mode,
        'contact_index_radix_min_contacts': radix_min_contacts,
        'contact_index_auto_min_bodies': 32,
        'contact_index_complexity': 'scan O(nbody*ncon); radix O((ncon+nbody)*log2(nbody+1))',
        'contact_index_cache': 'exact ncon + every enabled/ground event body key' if contact_topology_cache else 'disabled',
        'contact_index_cache_meta': '[valid,ncon] per world; key stores body or sentinel=nbody',
        'contact_index_selection': 'configurable heuristic; performance qualification pending',
        'scope': 'elliptic no-equality post-RNE and two site force/torque sensors',
        'uncovered': ['arbitrary sensor families','separate environment observation projection'],
        'blas_lapack': 'none: spatial vector/tree recurrences are not standard dense BLAS/LAPACK'}
