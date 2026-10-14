"""Cold packing recipe. Host outputs keep original insertion/copy semantics."""
from collections import OrderedDict
import numpy as np


class PackingRecipe:
    @classmethod
    def try_create(cls,plan):
        """Conservative first-pack eligibility; malformed/custom uses legacy.

        Inspect only known immutable descriptor types and builtin image dicts.
        Never change a legacy error's timing by raising during recipe selection.
        """
        from .schema import ObservationPlan,ObservableSpec
        try:
            if type(plan) is not ObservationPlan or type(plan.specs) is not tuple or type(plan.image_specs) is not tuple or type(plan.order) is not tuple:return None
            if any(type(s) is not ObservableSpec for s in plan.specs):return None
            for s in plan.specs:
                if type(s.name) is not str or type(s.modality) is not str or type(s.width) is not int or s.width<1:return None
                if type(s.enabled) is not bool or type(s.active) is not bool:return None
                if type(s.shape) is not tuple or any(type(d) is not int or d<0 for d in s.shape):return None
                if (int(np.prod(s.shape)) if s.shape else 1)!=s.width:return None
            for s in plan.image_specs:
                if type(s) is not dict or not all(k in s for k in ('name','shape','enabled','active')):return None
                if type(s['name']) is not str or type(s['shape']) is not tuple or any(type(d) is not int or d<0 for d in s['shape']):return None
                if type(s['enabled']) is not bool or type(s['active']) is not bool:return None
            if any(type(n) is not str for n in plan.order):return None
            recipe=cls(plan)
            return recipe if recipe.eligible else None
        except Exception:
            return None

    def __init__(self, plan):
        self.plan = plan
        self._legacy_packer = None
        declared=[s.name for s in plan.specs]+[s["name"] for s in plan.image_specs]
        # Public custom plans can collide lowdim/image names; original overwrite
        # and grouping rules are retained through its precise ordered packer.
        self.eligible=len(declared)==len(set(declared))
        offsets = plan.offsets
        specs = {s.name: s for s in plan.specs if s.enabled and s.active}
        images = {s["name"]: dict(s) for s in plan.image_specs if s["enabled"] and s["active"]}
        names = list(specs)+list(images)
        output_names = names
        if plan.order:
            output_names = list(OrderedDict.fromkeys(n for n in plan.order if n in specs or n in images))
        self.entries = tuple((n, specs.get(n), images.get(n), offsets.get(n)) for n in names)
        self.output_names = tuple(output_names)
        self.output_indices = tuple(names.index(n) for n in output_names)
        modalities = OrderedDict()
        for i in self.output_indices:
            _, spec, image, _ = self.entries[i]
            modalities.setdefault((spec.modality if spec else "image")+"-state", []).append(i)
        self.groups = tuple((n, tuple(ix)) for n, ix in modalities.items())
        self.image_signature = tuple((s["name"], tuple(s["shape"]), s["enabled"], s["active"]) for s in plan.image_specs)

    def matches(self, plan):
        try:
            return plan is self.plan and self.image_signature == tuple(
                (s["name"], tuple(s["shape"]), s["enabled"], s["active"]) for s in plan.image_specs)
        except Exception:
            return False

    def pack(self, values, images=None, concatenate_images=False):
        if not self.eligible:
            if self._legacy_packer is None:
                from .schema import pack_observations
                self._legacy_packer=pack_observations
            return self._legacy_packer(self.plan,values,images,concatenate_images)
        values = np.asarray(values)
        if values.shape != (self.plan.width,):
            raise ValueError(f"observed buffer shape {values.shape}, expected {(self.plan.width,)}")
        images = {} if images is None else images
        result = OrderedDict()
        arrays = []
        for name, spec, image, offset in self.entries:
            if spec is not None:
                data = values[offset:offset+spec.width].copy()
                value = data[0] if spec.shape == (1,) else data.reshape(spec.shape or ())
            else:
                value = np.asarray(images[name])
                if value.shape != tuple(image["shape"]):
                    raise ValueError(f"image shape mismatch: {name}")
            result[name] = value
            arrays.append(np.array([value]) if np.ndim(value) == 0 else np.asarray(value))
        result = OrderedDict((n, result[n]) for n in self.output_names)
        for modality, indices in self.groups:
            if modality == "image-state" and not concatenate_images:
                continue
            group = [arrays[i] for i in indices]
            # CPU paired measurement rejected manual per-slice assembly for
            # tiny plans. NumPy performs the exact copy/dtype/error semantics
            # with less Python work; the cold groups still avoid rebuilding it.
            result[modality] = np.concatenate(group, axis=-1)
        return result
