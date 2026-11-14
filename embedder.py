"""bge-m3 encoder shared by 07_build_index.py and 08_search.py.

One forward pass gives a dense 1024-d vector (meaning) and a sparse vector of
keyword weights (exact terms such as part names and spec numbers like 38.331).

--fake mode returns deterministic hash-based vectors so you can test the whole
pipeline (parsing -> Qdrant -> search) on a machine without a GPU or the model.
"""
import hashlib
import math
import re

DIM = 1024
MODEL = "BAAI/bge-m3"


class Encoder:
    def __init__(self, fake=False, max_length=1024, batch_size=64, device=None):
        self.fake, self.max_length, self.batch_size = fake, max_length, batch_size
        if not fake:
            from FlagEmbedding import BGEM3FlagModel
            kw = {"use_fp16": True}
            if device:
                kw["devices"] = device
            self.model = BGEM3FlagModel(MODEL, **kw)

    def encode(self, texts, max_length=None):
        """-> (dense: list[list[float]], sparse: list[(indices, values)])"""
        if self.fake:
            return [self._fake_dense(t) for t in texts], [self._fake_sparse(t) for t in texts]
        out = self.model.encode(texts, batch_size=self.batch_size, max_length=max_length or self.max_length,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense = [v.tolist() for v in out["dense_vecs"]]
        sparse = [([int(k) for k in w], [float(v) for v in w.values()]) for w in out["lexical_weights"]]
        return dense, sparse

    # ---- fake vectors for pipeline tests (no semantic quality!)
    @staticmethod
    def _tokens(t):
        return re.findall(r"[a-z0-9.]+", t.lower())

    def _fake_dense(self, t):
        v = [0.0] * DIM
        for tok in self._tokens(t):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 12) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _fake_sparse(self, t):
        w = {}
        for tok in self._tokens(t):
            k = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 250000
            w[k] = w.get(k, 0.0) + 0.1
        return list(w), list(w.values())
