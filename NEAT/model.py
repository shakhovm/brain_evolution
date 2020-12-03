import sys
import copy

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import importlib

from NEAT.genotype import Genotype

class Unit:
    def __init__(self, ref_node, num_in_features):
        self.ref_node = ref_node
        self.linear = self.build_linear(num_in_features)

    def set_weights(self, weights):
        if self.ref_node.get_layer() != 0:
            weights = torch.Tensor(weights).unsqueeze(0)
            for p in self.linear.parameters():
                p.data = weights

    def build_linear(self, num_in_features):
        if self.ref_node.get_layer() == 0:
            return None
        return nn.Linear(num_in_features, 1, False)

    def __str__(self):
        return 'Reference Node: ' + str(self.ref_node) + '\n'
    

class Phenotype(nn.Module):
    def __init__(self, genotype):
        super(Phenotype, self).__init__()
        self.genotype = genotype
        self.units = self.build_units()
        self.lin_modules = nn.ModuleList()
        self.activation = F.relu

        for unit in self.units:
            self.lin_modules.append(unit.linear)

    def forward(self, x):
        outputs = dict()
        input_units = [u for u in self.units if u.ref_node.get_layer() == self.genotype.get_layers().input()]
        output_units = [u for u in self.units if u.ref_node.get_layer() == self.genotype.get_layers().output()]
        stacked_units = copy.deepcopy(self.units)

        for u in input_units:
            outputs[u.ref_node.get_id()] = x[0][u.ref_node.get_id()]

        unit_idx = 0
        while len(stacked_units) > 0:
            current_unit = stacked_units.pop(0)

            if current_unit.ref_node.get_layer() != self.genotype.get_layers().input():
                inputs_ids = self.genotype.get_node_input_nodes(current_unit.ref_node.get_id())
                in_vec = autograd.Variable(torch.zeros((1, len(inputs_ids)), requires_grad=True))

                for i, input_id in enumerate(inputs_ids):
                    in_vec[0][i] = outputs[input_id]

                linear_module = self.lin_modules[unit_idx]
                
                if linear_module is not None:
                    scaled = linear_module(in_vec)
                    out = self.activation(scaled)
                else:
                    out = torch.zeros((1, 1))
                
                outputs[current_unit.ref_node.get_id()] = out
            unit_idx += 1
        output = autograd.Variable(torch.zeros((1, len(output_units)), requires_grad=True))
        for i, u in enumerate(output_units):
            output[0][i] = outputs[u.ref_node.get_id()]
        return output

    def build_units(self):
        units = []
        for n_id in self.genotype.topological_order():
            in_genes = self.genotype.get_in_edges(n_id)
            num_in = len(in_genes)
            weights = [g.get_weight() for g in in_genes]
            new_unit = Unit(self.genotype._node_by_id(n_id), num_in)
            new_unit.set_weights(weights)
            units.append(new_unit)
        return units
    
if __name__ == '__main__':
    gt = Genotype(2, 3, 1)
    gt.add_edge(0, 2)
    gt.add_edge(0, 3)
    gt.add_edge(0, 5)
    gt.add_edge(1, 3)
    gt.add_edge(1, 4)
    gt.add_edge(3, 5)
    gt.add_edge(2, 5)
    
    gt.add_rand_edge()
    gt.add_rand_node()
    
    for x in gt.get_nodes(): print(x)
    for x in gt.get_edges(): print(x)
    
    model = Phenotype(gt)
    res = model(torch.Tensor([3,4]).unsqueeze(0))
    print(res)