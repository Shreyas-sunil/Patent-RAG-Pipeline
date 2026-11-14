"""Hybrid search over the Qdrant index - import this from your RAG chatbot.

    from retrieve import Retriever
    r = Retriever()                       # loads bge-m3 (+ reranker if rerank=True)
    hits = r.search("Which clause defines RRC reconfiguration?", k=8, kind="standards")
    for h in hits: print(h["doc"], h["section"], h["score"])

Each hit is the chunk payload (doc, section, text, kind, area, title, date, cpc, ...)
plus a score. Dense and sparse results are fused with reciprocal rank fusion inside
Qdrant; the optional reranker then reorders the top 30.
"""
from qdrant_client import QdrantClient, models as m

from embedder import Encoder


class Retriever:
    def __init__(self, collection="corpus", url="http://localhost:6333", path=None,
                 rerank=False, fake=False):
        self.col = collection
        self.q = QdrantClient(path=path) if path else QdrantClient(url=url, timeout=60)
        self.enc = Encoder(fake=fake, max_length=512, batch_size=8)
        self.reranker = None
        if rerank and not fake:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def search(self, question, k=8, kind=None, area=None, doc=None, candidates=30):
        dense, sparse = self.enc.encode([question])
        conds = [m.FieldCondition(key=f, match=m.MatchValue(value=v))
                 for f, v in (("kind", kind), ("area", area), ("doc", doc)) if v]
        flt = m.Filter(must=conds) if conds else None
        prefetch = [m.Prefetch(query=dense[0], using="dense", limit=candidates + 10, filter=flt)]
        si, sv = sparse[0]
        if si:
            prefetch.append(m.Prefetch(query=m.SparseVector(indices=si, values=sv), using="sparse",
                                       limit=candidates + 10, filter=flt))
        res = self.q.query_points(self.col, prefetch=prefetch, query=m.FusionQuery(fusion=m.Fusion.RRF),
                                  limit=candidates, with_payload=True).points
        hits = [{**p.payload, "score": p.score} for p in res]
        if self.reranker and hits:
            scores = self.reranker.compute_score([[question, h["text"]] for h in hits], normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            for h, s in zip(hits, scores):
                h["score"] = float(s)
            hits.sort(key=lambda h: -h["score"])
        return hits[:k]
