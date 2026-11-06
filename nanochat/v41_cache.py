"""Request-owned bounded local + compressed global cache for V4.1 text models."""
import torch

from nanochat.v41_components import pack_fp4, unpack_fp4


class V41Cache:
    def __init__(self, config, batch_size, seq_len, device, dtype, bounded_replay=False):
        self.config, self.batch_size, self.max_seq_len = config, batch_size, seq_len
        self.device, self.dtype, self.bounded_replay = device, dtype, bounded_replay
        self.n_layers = config.n_layer
        self.globals, self.scales, self.index_k = {}, {}, {}
        for owner, ratio in config.sources().items():
            capacity = seq_len // ratio
            if config.fp4_cache:
                self.globals[owner] = torch.zeros(batch_size, capacity, config.head_dim // 2, device=device, dtype=torch.uint8)
                self.scales[owner] = torch.ones(batch_size, capacity, config.head_dim // 16, device=device).to(torch.float8_e4m3fn)
            else:
                self.globals[owner] = torch.zeros(batch_size, capacity, config.head_dim, device=device, dtype=dtype)
            self.index_k[owner] = torch.zeros(batch_size, capacity, config.index_dim, device=device, dtype=dtype)
        self.reset()

    def reset(self):
        self.pos = 0
        self.local, self.pending = {}, {}
        self.global_counts = {i: 0 for i in self.globals}
        self.layer_tokens = [0] * self.n_layers
        self.history = None

    def get_pos(self):
        return self.pos

    def append_global(self, owner, main, index):
        start = self.global_counts[owner]
        end = start + main.size(1)
        if end > self.globals[owner].size(1):
            raise ValueError("Global cache capacity exceeded")
        if self.config.fp4_cache:
            packed, scales = pack_fp4(main)
            self.globals[owner][:, start:end] = packed
            self.scales[owner][:, start:end] = scales
        else:
            self.globals[owner][:, start:end] = main
        self.index_k[owner][:, start:end] = index
        self.global_counts[owner] = end

    def global_values(self, owner):
        count = self.global_counts[owner]
        values = self.globals[owner][:, :count]
        if self.config.fp4_cache:
            values = unpack_fp4(values, self.scales[owner][:, :count]).to(self.dtype)
        return values, self.index_k[owner][:, :count]

    def append_local(self, layer, values):
        previous = self.local.get(layer)
        combined = values if previous is None else torch.cat([previous, values], 1)
        self.local[layer] = combined[:, -self.config.window_size:].clone()
        self.layer_tokens[layer] += values.size(1)
        return combined

    def storage_bytes(self):
        arrays = [*self.globals.values(), *self.scales.values(), *self.index_k.values(), *self.local.values()]
        arrays += [t for pair in self.pending.values() for t in pair]
        return sum(t.untyped_storage().nbytes() for t in arrays)

    def prefill(self, other):
        if self.pos or self.config != other.config or self.max_seq_len < other.pos:
            raise ValueError("Incompatible or nonempty destination cache")
        if other.batch_size not in (1, self.batch_size):
            raise ValueError("Cache batch cannot be broadcast")
        for owner, count in other.global_counts.items():
            self.globals[owner][:, :count] = other.globals[owner][:, :count]
            self.index_k[owner][:, :count] = other.index_k[owner][:, :count]
            if owner in self.scales:
                self.scales[owner][:, :count] = other.scales[owner][:, :count]
        def copy(t):
            return t.expand(self.batch_size, *t.shape[1:]).clone()
        self.local = {k: copy(v) for k, v in other.local.items()}
        self.pending = {k: (copy(v[0]), copy(v[1])) for k, v in other.pending.items()}
        self.history = None if other.history is None else copy(other.history)
        self.global_counts = other.global_counts.copy()
        self.layer_tokens = other.layer_tokens.copy()
        self.pos = other.pos
