from abc import ABC, abstractmethod
from random import choice, uniform, randint


class LayerManager:
    def __init__(self, num_layers):
        self._layers_n = num_layers
        
    def number(self):
        return self._layers_n
    
    def input(self):
        return 0
    
    def output(self):
        return self._layers_n - 1
    
    def add_one(self):
        self._layers_n += 1
    
    def random(self, lwr=0, upr=2):
        return randint(0, self._layers_n - 1)
    
    def successor(self, layer):
        return (layer + 1) % self._layers_n
    
    def greater(self, layer):
        layers = []
        while layer != self.output():
            layer = self.successor(layer)
            layers.append(layer)
        return layers
    
    