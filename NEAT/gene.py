from abc import ABC, abstractmethod
from random import choice, uniform, randint


class Gene(ABC):
    def __init__(self, identifier, mutable=True):
        self._id = identifier
        self._mutable = mutable
    
    @abstractmethod
    def _mutate(self):
        pass
    
    def get_id(self):
        return self._id
    
    def mutate(self):
        if self._mutable: self._mutate()
        

class NodeGene(Gene):
    def __init__(self, identifier, layer, mutable=True):
        super().__init__(identifier, mutable)
        self._layer = layer
    
    def __str__(self):
        layout = '[Node {}]:\tlayer = {}'
        return layout.format(self._id, self._layer)
    
    def _mutate(self): return None
    
    def get_layer(self): return self._layer


class EdgeGene(Gene):
    def __init__(self, identifier, nodes, node_from, node_to,
                 weight=1, active=True, mutable=True):
        super().__init__(identifier, mutable)
        self._nodes = nodes
        self._from = node_from
        self._to = node_to
        self._weight = weight
        self._active = True
    
    def __str__(self):
        layout = '[Edge {}]:\t{} -> {}\n\t\tw = {}\n\t\tactive = {}'
        return layout.format(self._id, self._from.get_id(), self._to.get_id(),
                             self._weight, self._active)
    
    def _mutate(self):
        candidates = filter(lambda x: x.get_layer() == self._to.get_layer(), self._nodes)
        candidates = list(candidates)
        self._to = choice(candidates)
        
    def get_weight(self): return self._weight
    
    def enable(self): self._active = True
    def disable(self): self._active = False
    def active(self): return self._active
