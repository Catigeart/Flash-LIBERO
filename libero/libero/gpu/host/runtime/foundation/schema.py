from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math

INT32_MAX = 2**31-1
SIZE_MAX = 2**64-1
DTYPE_BYTES = {"float64":8,"float32":4,"int64":8,"uint64":8,
               "int32":4,"uint32":4,"uint8":1,"int8":1,"bool":1}

def checked_product(values, *, limit=INT32_MAX):
    result=1
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative Python integer dimensions required")
        result *= value
        if result > limit:
            raise OverflowError(f"index/byte product {result} exceeds {limit}")
    return result

def content_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()

def c_strides(shape, itemsize):
    stride=itemsize
    result=[]
    for extent in reversed(shape):
        result.append(stride)
        stride*=extent
    return tuple(reversed(result))

@dataclass(frozen=True)
class FieldDescriptor:
    name: str
    dtype: str
    shape: tuple[int,...]
    owner: str
    lifetime: str
    epoch: str
    writers: tuple[str,...]
    valid_write: str
    read_domain: str
    retain: str
    offset: int=0
    sentinel_unread: bool=False
    access: str="read_write"

    def __post_init__(self):
        if self.access not in ("read","write","read_write"):raise ValueError("explicit read/write access required")
        if type(self.shape)!=tuple or type(self.offset)!=int:raise ValueError("immutable shape and integer byte offset required")
        if not all(isinstance(x,str) and x for x in (self.valid_write,self.read_domain,self.retain)):raise ValueError("write/read/retain source domains required")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError("unsupported native dtype")
        if not self.name or not self.owner or not self.epoch or not self.lifetime:
            raise ValueError("full field provenance required")
        if not self.writers or len(set(self.writers))!=len(self.writers):
            raise ValueError("unique explicit per-stage writers required")
        if self.offset < 0 or self.offset % DTYPE_BYTES[self.dtype]:
            raise ValueError("misaligned field offset")
        checked_product(self.shape)
        checked_product([self.physical_elements,DTYPE_BYTES[self.dtype]],limit=SIZE_MAX)
        if self.logical_elements==0 and not self.sentinel_unread:
            raise ValueError("zero logical domain must declare backend sentinel unread")

    @property
    def logical_elements(self):
        return math.prod(self.shape)

    @property
    def physical_elements(self):
        return max(1,self.logical_elements)

    @property
    def physical_shape(self):
        return self.shape if self.logical_elements else (1,)

    @property
    def strides(self):
        return c_strides(self.physical_shape,DTYPE_BYTES[self.dtype])

    def to_json(self):
        out=asdict(self)
        out.update(logical_elements=self.logical_elements,
                   physical_shape=self.physical_shape,strides_bytes=self.strides,
                   allocation_bytes=self.physical_elements*DTYPE_BYTES[self.dtype],
                   layout="C contiguous, world major")
        return out
