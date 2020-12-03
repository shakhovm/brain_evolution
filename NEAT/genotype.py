from abc import ABC, abstractmethod
from random import choice, uniform, randint
from copy import deepcopy

from NEAT.layermanager import LayerManager
from NEAT.gene import NodeGene, EdgeGene


class Genotype:
    def __init__(self, num_in, num_hidden, num_out):
        self._lm = LayerManager(3)
        self.__nodes_n = 0
        self.__edges_n = 0
        self._nodes = []
        self._edges = []
        
        self.add_nodes(num_in, self._lm.input())
        self.add_nodes(num_hidden, 1)
        self.add_nodes(num_out, self._lm.output())
    
    def _increase_layers_from(self, layer):
        for node in self._nodes:
            if node.get_layer() >= layer:
                node._layer += 1
        self._lm.add_one()
    
    def _node_by_id(self, identifier):
        return next(x for x in self._nodes if x.get_id() == identifier)
    
    def _nodes_by_layer(self, layer):
        nodes = filter(lambda x: x.get_layer() == layer, self._nodes)
        return [x.get_id() for x in nodes]
    
    def _nodes_by_layers(self, layers):
        nodes = []
        for l in layers:
            nodes.extend(self._nodes_by_layer(l))
        return nodes
            
    def _edge_by_ends_ids(self, id_from, id_to):
        return next((x for x in self._edges if x._from.get_id() == id_from and
                                                x._to.get_id() == id_to), None)
    
    def _valid_edges(self, identifier):
        node = self._node_by_id(identifier)
        if node.get_layer() == self._lm.output(): return list()
        candidates = self._nodes_by_layers(self.get_layers().greater(node.get_layer()))
        return [x for x in candidates if not self.edge_exists(identifier, x)]
    
    def node_index_in_layer(self, indentifier, layer):
        layer_nodes = self._nodes_by_layer(layer)
        layer_nodes = sorted(layer_nodes)
        return layer_nodes.index(indentifier)
    
    def get_in_edges(self, identifier):
        edges = []
        for edge in self.get_edges():
            if edge.active() and edge._to.get_id() == identifier:
                edges.append(edge)
        return edges
    
    def get_node_output_nodes(self, identifier):
        return [x._to.get_id() for x in self.get_active_edges() if x._from.get_id() == identifier]
    
    def get_node_input_nodes(self, identifier):
        return [x._from.get_id() for x in self.get_active_edges() if x._to.get_id() == identifier]
    
    def get_node_ids(self):
        return [x.get_id() for x in self.get_nodes()]
    
    def get_layers(self): return self._lm
    def get_nodes(self): return self._nodes
    def get_edges(self): return self._edges
    
    def get_active_edges(self):
        return list(filter(lambda x: x.active(), self.get_edges()))
    
    def edge_exists(self, id_from, id_to):
        edge = self._edge_by_ends_ids(id_from, id_to)
        return edge is not None and edge.active()
        
    def add_node(self, layer):
        self._nodes.append(NodeGene(self.__nodes_n, layer))
        self.__nodes_n += 1
        return self.__nodes_n - 1
    
    def add_nodes(self, n, layer):
        for i in range(n):
            self.add_node(layer)
        
    def add_edge(self, id_from, id_to, weight=1):
        node_from = self._node_by_id(id_from)
        node_to = self._node_by_id(id_to)
        self._edges.append(EdgeGene(self.__edges_n, self._nodes, node_from, node_to, weight))
        self.__edges_n += 1
        
    def add_rand_edge(self):
        iters_limit = 10
        w = uniform(0, 1)
        layer_from = self.get_layers().random(0, self.get_layers().output() - 1)
        from_cands = self._nodes_by_layer(layer_from)
        
        id_cand = choice(from_cands)
        to_cands = self._valid_edges(id_cand)
        while iters_limit and not to_cands:
            id_cand = choice(from_cands)
            to_cands = self._valid_edges(id_cand)
            iters_limit -= 1
        
        if not to_cands: return None
        self.add_edge(id_cand, choice(to_cands), w)
        
    def add_rand_node(self):
        layer_from = self.get_layers().random(0, self.get_layers().output() - 1)
        id_from = self._nodes_by_layer( layer_from )
        id_to = self._nodes_by_layers(self.get_layers().greater(layer_from))
        
        cands = []
        for f,t in zip(id_from, id_to):
            if self.edge_exists(f,t): cands.append((f,t))
        
        if not cands: return None
        f,t = choice(cands)
        edge = self._edge_by_ends_ids(f,t)
        edge.disable()
        
        f_layer = edge._from.get_layer()
        t_layer = edge._to.get_layer()
        mid_layer = f_layer + (t_layer - f_layer) // 2
        if (t_layer - f_layer) % 2 != 0:
            mid_layer += 1
            print(f_layer, mid_layer, t_layer + 1)
            self._increase_layers_from(mid_layer)
        
        node_id = self.add_node(mid_layer)
        self.add_edge(f, node_id, edge._weight)
        self.add_edge(node_id, t)
        
    def mutate_rand_edge(self):
        choice(self.get_edges()).mutate()
        
    def mutate_rand_weight(self):
        choice(self.get_edges())._weight = uniform(0, 1)
        
    def crossover(self, other):
        other_cp = deepcopy(other)
        return other_cp
    
    def _visit_node(self, identifier, visited, ordered):
        if not visited[identifier]:
            visited[identifier] = True
            for out_id in self.get_node_output_nodes(identifier):
                self._visit_node(out_id, visited, ordered)
            ordered.append(identifier)
    
    def topological_order(self):
        ordered = []
        visited = [False] * len(self.get_nodes())
        for node_id in self.get_node_ids():
            self._visit_node(node_id, visited, ordered)
        return ordered[::-1]

if __name__ == '__main__':
    gt = Genotype(2, 3, 1)
    
    gt.add_edge(0, 2)
    gt.add_edge(0, 3)
    gt.add_edge(0, 5)
    gt.add_edge(1, 3)
    gt.add_edge(1, 4)
    gt.add_edge(3, 5)
    gt.add_edge(2, 5)
    
    gt.mutate_rand_weight()
    
    for x in gt.get_nodes(): print(x)
    print(x)
    for x in gt.get_edges(): print(x)

    
    
    
    