"""Exact captured source generations, independent of future device uploads."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from .contracts import ContractError, ModelSpec, _hash
from .census_registry import describe_census_record


def captured_generation(record_path, diff_path, review_path, review_sha256):
    """Return canonical base/post specs only for this independently reviewed capture.

    Census descriptors use raw-byte array SHA descriptors. Factory decoder readset
    digests use another encoding and must still be checked by their own holder.
    This function does not grant a kernel license or authorize a future mutation.
    """
    review_raw = Path(review_path).read_bytes()
    if hashlib.sha256(review_raw).hexdigest() != review_sha256:
        raise ContractError('generation review content changed')
    review = json.loads(review_raw)
    if review.get('failures') or not review.get('checks'):
        raise ContractError('generation review did not pass')
    diff_raw = Path(diff_path).read_bytes()
    diff = json.loads(diff_raw)
    matches = [r for r in review['rows'] if r['task'] == diff['task']]
    if len(matches) != 1:
        raise ContractError('generation review task identity missing or ambiguous')
    row = matches[0]
    if row['diff_sha256'] != hashlib.sha256(diff_raw).hexdigest():
        raise ContractError('captured generation diff content changed')
    if (row['base_sha256'] != diff['base_file_sha256'] or
            row['post_sha256'] != diff['post_file_sha256']):
        raise ContractError('generation model identities differ from review')
    required = ('options_equal', 'static_scalars_equal', 'topology_equal',
                'source_generation_qualified')
    if any(diff.get(key) is not True for key in required):
        raise ContractError('generation source constraints not satisfied')
    if not diff['source_generation_checks'] or any(v is not True for v in diff['source_generation_checks'].values()):
        raise ContractError('generation typed-field rule failed')
    post, descriptor = describe_census_record(record_path)
    task = descriptor['task']
    if diff['task'] != task['suite'] + '/' + task['task_name'] or post.model_sha256 != row['post_sha256']:
        raise ContractError('generation task or post model mismatch')
    record = json.loads(Path(record_path).read_bytes())
    facts = json.loads(Path(record['facts_artifacts']['post_reset_facts.json']['path']).read_bytes())
    arrays = copy.deepcopy(facts['actual_model_arrays'])
    fields = tuple(sorted(change['field'] for change in diff['changed_arrays']))
    if fields != tuple(sorted(row['changedfields'])) or len(fields) != len(set(fields)):
        raise ContractError('generation changed-field set differs from review')
    for change in diff['changed_arrays']:
        field = change['field']
        if field not in ('body_pos', 'body_quat', 'site_rgba'):
            raise ContractError('unqualified captured generation field')
        actual = arrays[field]
        if (actual['sha256'] != change['post_raw_sha256'] or
                actual['shape'] != change['post_shape'] or actual['dtype'] != change['post_dtype'] or
                change['base_shape'] != change['post_shape'] or change['base_dtype'] != change['post_dtype']):
            raise ContractError('generation array digest or layout mismatch')
        actual['sha256'] = change['base_raw_sha256']
    metadata = post.to_factory_metadata()
    metadata['model_sha256'] = row['base_sha256']
    metadata['static_arrays_sha256'] = _hash({'arrays': arrays, 'scalars': facts['actual_model_scalars']})
    base = ModelSpec.from_factory_metadata(metadata, features=post.features)
    if base.specialization_key != post.specialization_key:
        raise ContractError('captured generation changed structural specialization')
    relation = {'schema': 'captured-source-generation-v1', 'task': diff['task'],
                'base_metadata': base.to_factory_metadata(), 'post_metadata': post.to_factory_metadata(),
                'features': list(post.features), 'base_identity': base.identity_key,
                'derived_identity': post.identity_key, 'changed_fields': list(fields),
                'base_model_path': diff['base_mjb'], 'post_model_path': diff['post_mjb'],
                'review_path': str(review_path), 'review_sha256': review_sha256,
                'diff_path': str(diff_path), 'diff_sha256': row['diff_sha256'],
                'digest_scheme': 'census-raw-byte-array-descriptors-v1',
                'source_generation_qualified': True, 'kernel_license_granted': False,
                'future_runtime_upload_qualified': False}
    return base, post, relation


def load_captured_identity_join(review_path, review_sha256, task):
    """Resolve one already reviewed four-identity capture; never license a kernel.

    The caller pins the independent review SHA. An old model outside the capture
    is rejected even when its dimensions agree. Device epochs are not published.
    """
    raw = Path(review_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != review_sha256:
        raise ContractError('identity join review content changed')
    review = json.loads(raw)
    if review.get('schema') != 'Independent digest scheme source join review.v1' or review.get('failures') or review.get('tasks') != 130:
        raise ContractError('identity join independent review not qualified')
    pin = review['report_pin']
    joined_raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(joined_raw).hexdigest() != pin['sha256']:
        raise ContractError('identity join captured matrix changed')
    joined = json.loads(joined_raw)
    rows = [row for row in joined['models'] if row['task'] == task]
    if len(rows) != 1:
        raise ContractError('exact task identity not found in captured matrix')
    row = rows[0]
    capture = row['captured_generation']
    generation_pin = review['source_generation_review_pin']
    if generation_pin['path'] != capture['review_path'] or generation_pin['sha256'] != capture['review_sha256']:
        raise ContractError('source generation review differs from identity join')
    generation_raw = Path(generation_pin['path']).read_bytes()
    if hashlib.sha256(generation_raw).hexdigest() != generation_pin['sha256']:
        raise ContractError('source generation independent review changed')
    specs = {}
    for scheme in ('native', 'census'):
        for stage in ('base', 'post'):
            key = scheme + '_' + stage
            metadata = row[key + '_metadata'] if scheme == 'native' else capture[stage + '_metadata']
            spec = ModelSpec.from_factory_metadata(metadata, features=capture['features'])
            identity = row[key + '_identity']
            if spec.identity_key != identity:
                raise ContractError('captured ModelSpec identity changed')
            if spec.model_sha256 != capture[stage + '_metadata']['model_sha256']:
                raise ContractError('captured scheme models differ')
            specs[key] = spec
    return specs, copy.deepcopy(row)


def model_spec_for_native_metadata(metadata, review_path, review_sha256, task):
    """Bind exact native decoder metadata to reviewed feature tokens for one task.

    Arbitrary features=() cannot stand in for the actual structural feature
    descriptor. Only captured raw-base/post identities resolve here; a future
    model generation requires its own explicit source qualification.
    """
    specs, row = load_captured_identity_join(review_path, review_sha256, task)
    matches = [spec for key, spec in specs.items() if key.startswith('native_')
               and spec.model_sha256 == metadata.get('model_sha256')]
    if not matches:
        raise ContractError('native model outside exact captured task identities')
    resolved = ModelSpec.from_factory_metadata(metadata, features=row['captured_generation']['features'])
    if not any(resolved.identity_key == expected.identity_key for expected in matches):
        raise ContractError('native dimensions/options/static/layout differ from captured identity')
    return resolved


def derive_native_parameter_spec(base_spec, base_metadata, derived_metadata, *, declared_fields):
    """Describe a freshly decoded parameter candidate, without production grant.

    Payload source recipe, fixture IDs, shapes, finite values and quaternion
    semantics must be validated before native/device writes by the caller. This
    checks the complete decoder readset change and never asserts device ready.
    """
    from .contracts import _digest
    fields = tuple(declared_fields)
    if not fields or len(fields) != len(set(fields)) or set(fields) - set(base_spec.mutable_parameters):
        raise ContractError('parameter fields outside explicit model declaration')
    if set(fields) - {'body_pos', 'body_quat'}:
        raise ContractError('physics parameter derivation does not consume visual changes')
    actual_base = ModelSpec.from_factory_metadata(base_metadata, features=base_spec.features)
    if actual_base.identity_key != base_spec.identity_key:
        raise ContractError('parameter base metadata does not match source identity')
    if base_metadata['static_array_inventory'] != derived_metadata['static_array_inventory']:
        raise ContractError('parameter update changed static array layouts')
    before = base_metadata['array_sha256']
    after = derived_metadata['array_sha256']
    if set(before) != set(after):
        raise ContractError('parameter update changed source readset fields')
    for field, digest in after.items():
        _digest(digest, 'parameter candidate array digest')
    changed = tuple(sorted(name for name in before if before[name] != after[name]))
    if set(changed) - set(fields):
        raise ContractError('parameter update changed undeclared immutable arrays')
    derived = ModelSpec.from_factory_metadata(derived_metadata, features=base_spec.features)
    if derived.specialization_key != base_spec.specialization_key:
        raise ContractError('parameter update changed dimensions/options/topology/features/layout')
    if derived.mutable_parameters != base_spec.mutable_parameters:
        raise ContractError('parameter update changed field permissions')
    if changed and (derived.static_arrays_sha256 == base_spec.static_arrays_sha256 or derived.model_sha256 == base_spec.model_sha256):
        raise ContractError('changed arrays require fresh actual model and static digests')
    if not changed and derived.identity_key != base_spec.identity_key:
        raise ContractError('unchanged arrays cannot explain a different model identity')
    return derived, {'schema': 'native-parameter-diagnostic-preparation-v1',
                     'base_identity': base_spec.identity_key,
                     'derived_identity': derived.identity_key,
                     'declared_fields': list(fields), 'changed_fields': list(changed),
                     'actual_payload_source_qualified': False,
                     'device_upload_qualified': False, 'device_ready_published': False,
                     'numerical_license_granted': False}



def prepare_parameter_descriptor(model_generation, update, derived_spec, *, metadata_preparation=None):
    """Preflight a descriptor transition before the actual device transaction.

    Returns a private plan tuple. The source owners still qualify/prepare their
    actual readsets and the device publication separately.
    """
    if (model_generation.spec.identity_key != update.model_identity or
            model_generation.generation != update.base_generation or
            model_generation.parameters_sha256 != update.base_digest):
        raise ContractError('parameter descriptor token no longer matches base')
    if (derived_spec.specialization_key != model_generation.spec.specialization_key or
            derived_spec.mutable_parameters != model_generation.spec.mutable_parameters):
        raise ContractError('parameter descriptor changed immutable structure/permissions')
    if derived_spec.identity_key != update.model_identity and not update.fields:
        raise ContractError('new descriptor requires a nonempty parameter update')
    if derived_spec.identity_key != update.model_identity:
        if not metadata_preparation or metadata_preparation.get('schema') != 'native-parameter-diagnostic-preparation-v1':
            raise ContractError('fresh descriptor requires complete native metadata preflight')
        if (metadata_preparation['base_identity'] != update.model_identity or metadata_preparation['derived_identity'] != derived_spec.identity_key or set(metadata_preparation['declared_fields']) != set(update.fields)):
            raise ContractError('descriptor metadata preflight differs from update token')
    return update, derived_spec, copy.deepcopy(metadata_preparation)


def commit_parameter_descriptor(model_generation, prepared):
    """Commit token against old identity, then rebind the same generation object.

    Call only after actual native/device transaction success, holding the caller
    lock. Never replace spec before token commit. On device failure quarantine;
    this function does not restore a partially uploaded native/device model.
    """
    update, derived_spec, metadata_preparation = prepared
    prepare_parameter_descriptor(model_generation, update, derived_spec, metadata_preparation=metadata_preparation)
    model_generation.commit_parameter_update(update)
    model_generation.spec = derived_spec
    return model_generation.generation

