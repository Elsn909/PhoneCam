"""CUDA Graphs : un calcul de forme fixe est enregistré une fois, puis rejoué en un seul lancement.

Les réseaux de RVC enchaînent des centaines de petites opérations : sans graphe, la carte attend
surtout le processeur. Même principe que tools/cuda_graph.py de RVC 2.3 (licence MIT), réduit à
l'essentiel. En cas d'échec de capture, on retombe simplement sur le calcul normal.
"""
import torch


class Graphs:
    def __init__(self, enabled=True, keep=6):
        self.enabled = enabled and torch.cuda.is_available()
        self.keep = keep
        self.cache = {}
        self.failed = set()

    def clear(self):
        self.cache.clear()

    def run(self, key, fn, *inputs):
        if not self.enabled:
            return fn(*inputs)
        sig = (key,) + tuple((tuple(t.shape), t.dtype) for t in inputs)
        if sig in self.failed:
            return fn(*inputs)
        entry = self.cache.get(sig)
        if entry is None:
            try:
                entry = self._capture(fn, inputs)
            except Exception:  # noqa: BLE001
                self.failed.add(sig)
                torch.cuda.synchronize()
                return fn(*inputs)
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            self.cache[sig] = entry
        static_in, graph, static_out = entry
        for s, v in zip(static_in, inputs):
            s.copy_(v, non_blocking=True)
        graph.replay()
        return tuple(o.clone() for o in static_out) if isinstance(static_out, tuple) else static_out.clone()

    @staticmethod
    def _capture(fn, inputs):
        static_in = tuple(t.clone() for t in inputs)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                fn(*static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # « thread_local » : la vidéo (ONNX Runtime, PyTorch) continue d'utiliser la carte dans ses
        # propres fils pendant l'enregistrement, sans erreur ni d'un côté ni de l'autre.
        with torch.no_grad(), torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = fn(*static_in)
        torch.cuda.synchronize()
        return static_in, graph, out
