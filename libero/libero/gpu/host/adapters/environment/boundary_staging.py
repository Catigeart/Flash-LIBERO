"""CPU02 scoped pinned selected-world D2H helper; Warp backend injected."""
import operator

class BoundaryRangeStaging:
    def __init__(self, backend, stream):
        self.backend, self.stream = backend, stream
        self.buffers = {}
        self.bytes_copied = 0

    def read_many(self, requests):
        """requests=(slot, owner, offset, count), exact contiguous scalar owners.

        Caller pins field owners, serializes mutation and performs first-error
        validation. All requests are checked before copying. Results own memory.
        """
        prepared = []
        seen = set()
        for slot, array, offset, count in requests:
            offset, count = operator.index(offset), operator.index(count)
            if slot in seen:
                raise ValueError('duplicate boundary staging slot')
            seen.add(slot)
            if not array.is_contiguous or offset < 0 or count < 0 or offset + count > array.size:
                raise ValueError('invalid contiguous boundary range')
            if array.device != self.stream.device:
                raise ValueError('boundary owner/stream device mismatch')
            target = self.buffers.get(slot)
            if target is None or target.dtype != array.dtype or target.size < count:
                target = self.backend.empty(count, dtype=array.dtype, device='cpu', pinned=True)
                # One grow-only buffer per declared slot, not one per ncon.
                self.buffers[slot] = target
            prepared.append((slot, array, offset, count, target))
        for _, array, offset, count, target in prepared:
            # Warp copy(count=0) means entire source, NOT an empty range.
            if count:
                self.backend.copy(target, array, src_offset=offset, count=count, stream=self.stream)
        if any(count for _, _, _, count, _ in prepared):
            self.backend.synchronize_stream(self.stream)
        results = {slot: target.numpy().reshape(-1)[:count].copy() for slot, _, _, count, target in prepared}
        self.bytes_copied += sum(value.nbytes for value in results.values())
        return results

