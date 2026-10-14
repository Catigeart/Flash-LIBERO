"""Original camera cache and task-label semantics, independent of device storage."""
from collections.abc import Mapping
import numpy as np


def image_descriptor(spec):
    name=spec["name"]
    for suffix in ("segmentation_instance","segmentation_class","segmentation_element","image","depth"):
        tail="_"+suffix
        if name.endswith(tail):
            camera=name[:-len(tail)]
            if spec.get("camera",camera)!=camera or spec.get("kind",suffix)!=suffix:
                raise ValueError("camera descriptor does not match observable name")
            return camera,suffix
    raise NotImplementedError("unknown camera observable: "+name)


def segmentation_mapping(descriptors,mappings):
    mappings={} if mappings is None else mappings
    result={}
    for _,kind in descriptors:
        if kind in ("segmentation_instance","segmentation_class"):
            group=kind[len("segmentation_"):]
            if group not in mappings:raise ValueError("missing task segmentation mapping: "+group)
            mapping=mappings[group]
            if not isinstance(mapping,Mapping):raise ValueError("invalid task segmentation mapping")
            normalized={int(k):int(v) for k,v in mapping.items()}
            if any(k<0 or v<0 for k,v in normalized.items()):raise ValueError("negative task segmentation mapping")
            result[group]=normalized
    return result


def render_sampled_images(specs,descriptors,mappings,layout,scenes,generations,rendered,images,
                          bridges,image_convention,scene_option,parent_generation=None):
    """Render only new sample events; preserve RGB's shared raw-depth cache."""
    by_name={spec["name"]:i for i,spec in enumerate(specs)}
    depth_results={}
    for i,(spec,(camera,kind)) in enumerate(zip(specs,descriptors)):
        if not spec["enabled"] or generations[i]==rendered[i]:continue
        h,w,_=spec["shape"]
        bridge=bridges[(h,w)] if isinstance(bridges,Mapping) else bridges
        if kind=="depth":
            # The device capture retains the RGB scene that existed at this
            # depth sensor's own sampling event, including different rates.
            rgb_index=by_name.get(camera+"_image")
            if rgb_index is None:raise ValueError("depth observable has no RGB cache producer")
            parent=int(generations[rgb_index]) if parent_generation is None else int(parent_generation[i])
            if parent==0:
                images[i]=np.zeros((h,w,1))
            elif camera in depth_results and depth_results[camera][0]==parent:
                images[i]=np.expand_dims(depth_results[camera][1],axis=-1)
            else:
                fields={name:scenes[i,offset:offset+int(np.prod(shape))].reshape(shape)
                        for name,(offset,shape) in layout.items()}
                bridge.publish(fields)
                _,depth=bridge.render_rgb(camera,image_convention,scene_option,depth=True)
                images[i]=np.expand_dims(depth,axis=-1)
            rendered[i]=generations[i]
            continue
        fields={name:scenes[i,offset:offset+int(np.prod(shape))].reshape(shape)
                for name,(offset,shape) in layout.items()}
        bridge.publish(fields)
        if kind=="image":
            di=by_name.get(camera+"_depth")
            if di is not None:
                if tuple(specs[di]["shape"][:2])!=(h,w):raise ValueError("depth/RGB cache dimensions differ")
                rgb,depth=bridge.render_rgb(camera,image_convention,scene_option,depth=True)
                images[i]=rgb
                depth_results[camera]=(int(generations[i]),depth)
            else:images[i]=bridge.render_rgb(camera,image_convention,scene_option)
        else:
            raw=bridge.render_segmentation(camera,image_convention,scene_option)
            seg=np.expand_dims(raw[:,:,1],axis=-1)
            group=kind[len("segmentation_"):]
            if group!="element":
                mapping=mappings[group]
                # Exact robosuite mapping: unknown -> -1 then +1, background zero.
                seg=np.fromiter((mapping.get(int(x),-1) for x in seg.flat),dtype=np.int32,count=seg.size).reshape(h,w,1)+1
            images[i]=seg
        rendered[i]=generations[i]
