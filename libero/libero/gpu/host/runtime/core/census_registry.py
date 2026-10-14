"""Build exact model descriptors from census facts, without physics or GPU."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .contracts import ContractError, ModelSpec, _hash


FAMILY_FIELDS = ('body_parentid', 'body_dofadr', 'body_dofnum', 'body_jntadr',
                 'body_jntnum', 'dof_Madr', 'dof_bodyid', 'dof_jntid', 'dof_parentid',
                 'jnt_bodyid', 'jnt_dofadr', 'jnt_qposadr', 'jnt_type',
                 'actuator_trnid', 'actuator_trntype', 'actuator_dyntype',
                 'actuator_gaintype', 'actuator_biastype', 'sensor_type',
                 'sensor_needstage', 'sensor_dim', 'sensor_adr')


def feature_tokens(facts):
    tokens = []
    features = facts['support_features']
    for category, value in features.items():
        if isinstance(value, dict) and 'values' in value:
            if not value.get('available', False):
                tokens.append(f'{category}:unavailable')
            for row in value['values']:
                if row.get('count', 0):
                    tokens.append(f'{category}:{row.get("name") or row["value"]}')
        elif category == 'numeric_nonzero_counts':
            tokens.extend(f'{key}:nonzero' for key, count in value.items() if count)
        elif isinstance(value, bool):
            if value:
                tokens.append(category)
        elif isinstance(value, int) and value:
            tokens.append(f'{category}:nonzero')
    options = facts['options']['scalars']
    nv = facts['dimensions']['nv']
    jacobian = int(options['jacobian'])
    resolved = 'dense' if jacobian == 0 or (jacobian == 2 and nv < 60) else 'sparse'
    tokens.extend((f'jacobian:{resolved}', f'cone:{options["cone"]}',
                   f'solver:{options["solver"]}', f'integrator:{options["integrator"]}'))
    return tuple(sorted(set(tokens)))


def describe_census_record(record_path):
    record_path = Path(record_path)
    raw = record_path.read_bytes()
    record = json.loads(raw)
    if record.get('status') != 'success':
        raise ContractError('census model construction did not succeed')
    entry = record['facts_artifacts']['post_reset_facts.json']
    facts_path = Path(entry['path'])
    facts_raw = facts_path.read_bytes()
    if hashlib.sha256(facts_raw).hexdigest() != entry['sha256']:
        raise ContractError('census exact-facts content differs from evidence index')
    facts = json.loads(facts_raw)
    # Record.options is a lossy display summary in the first census revision.
    # Always use native scalar values and array content descriptors from facts.
    options = dict(facts['options']['scalars'])
    options['array_descriptors'] = facts['options']['arrays']
    options['callbacks_sha256'] = facts['callback_identity_sha256']
    dims = dict(facts['dimensions'])
    dims.update({name: value for name, value in facts['actual_model_scalars'].items()
                 if name.startswith('n') and isinstance(value, int) and not isinstance(value, bool)})
    static_digest = _hash({'arrays': facts['actual_model_arrays'], 'scalars': facts['actual_model_scalars']})
    metadata = {'dimensions': dims, 'options': options,
                'model_sha256': record['artifacts']['model.mjb']['sha256'],
                'topology_sha256': facts['topology_identity_sha256'],
                'static_arrays_sha256': static_digest}
    features = feature_tokens(facts)
    spec = ModelSpec.from_factory_metadata(metadata, features=features)
    family = _hash({'joint_layout': {name: facts['actual_model_arrays'].get(name) for name in FAMILY_FIELDS},
                    'dimensions': spec.dimensions.as_dict(), 'features': features,
                    'solver_options': {name: options[name] for name in
                                       ('solver', 'integrator', 'cone', 'jacobian', 'enableflags', 'disableflags')}})
    descriptor = {'schema': 'census-model-descriptor-v1', 'task': record['task'],
                  'record_path': str(record_path), 'record_sha256': hashlib.sha256(raw).hexdigest(),
                  'model_path': record['artifacts']['model.mjb']['path'],
                  'metadata': metadata, 'features': list(features),
                  'identity_key': spec.identity_key, 'specialization_key': spec.specialization_key,
                  'family_group_key': family,
                  'family_group_key_is_numeric_license': False,
                  'native_configuration_sha256': facts['model_configuration_identity_sha256'],
                  'post_reset_configuration_sha256': record['post_reset_model_configuration_identity_sha256'],
                  'post_reset_changed_model_fields': record['post_reset_changed_model_array_fields'],
                  'required_components': {component: {'candidate_id': None, 'status': 'pending',
                                                      'reason': 'source-qualified candidate and independent numerical report required'}
                                          for component in ('F', 'D', 'C', 'S', 'N', 'Euler', 'observables')},
                  'options_source': 'post_reset_facts.json exact scalar and array descriptors',
                  'descriptor_stage': 'post_reset_saved_mjb',
                  'base_raw_compile_is_same_model_identity': False,
                  'kernel_license_granted': False}
    return spec, descriptor


def build_registry(census_root, output_path):
    descriptors, failures = [], []
    for record_path in sorted(Path(census_root).glob('shard*/full/task_*/record.json')):
        try:
            _, descriptor = describe_census_record(record_path)
            descriptors.append(descriptor)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            failures.append({'record_path': str(record_path), 'error': str(exc)})
    families = {}
    for row in descriptors:
        families.setdefault(row['family_group_key'], []).append(row['task'])
    result = {'schema': 'all-task-model-admission-matrix-v1', 'models': descriptors,
              'model_count': len(descriptors), 'family_groups': families,
              'family_group_count': len(families), 'descriptor_failures': failures,
              'numerically_accepted_models': 0,
              'scope': 'metadata admission foundation, all fresh model numerical qualifications pending'}
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result

