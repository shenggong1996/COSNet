import torch
import torch.nn as nn
import torch.nn.functional as F
from model.roost import CompositionNetwork, SimpleNetwork, ResidualNetwork
from model.cgcnn import StructureNetwork

class MultiModalNetwork(nn.Module):
    """
    Create a multimodal network that combines representations of roost and de-cgcnn and makes prediction
    """
    def __init__(self, structure_net_params, composition_net_params, 
                 weight_net_params, output_net_params, element_sum):
        super().__init__()
        self.element_sum = element_sum
        self.structure_network = StructureNetwork(structure_net_params)
        self.composition_network = CompositionNetwork(composition_net_params)
        self.structure_weight_network = SimpleNetwork(weight_net_params["input_dim"],
                                                      weight_net_params["output_dim"],
                                                      weight_net_params["hidden_layer_dims"])
        self.composition_weight_network = SimpleNetwork(weight_net_params["input_dim"],
                                                        weight_net_params["output_dim"],
                                                        weight_net_params["hidden_layer_dims"])
        if element_sum:
            self.combined_representation_update_network = SimpleNetwork(weight_net_params["input_dim"],
                                                        weight_net_params["input_dim"],
                                                        [weight_net_params["input_dim"]])
        else:
            self.combined_representation_update_network = SimpleNetwork(weight_net_params["input_dim"]*2,
                                                        weight_net_params["input_dim"],
                                                        [weight_net_params["input_dim"]])
        self.output_network = ResidualNetwork(output_net_params["input_dim"],
                                                        output_net_params["output_dim"],
                                                        output_net_params["hidden_layer_dims"])
        self.non_negative_1 = nn.ReLU()
        self.non_negative_2 = nn.Softplus()
    
    def weight_determination(self, composition_representation, structure_representation, structure_presence):
        structure_weight = self.structure_weight_network(structure_representation)
        composition_weight = self.composition_weight_network(composition_representation)
        structure_weight = self.non_negative_2(structure_weight)*structure_presence #if structure is not present, then structure weight is zero
        composition_weight = self.non_negative_2(composition_weight)
        structure_weight_final = torch.div(structure_weight, structure_weight+composition_weight)
        composition_weight_final = torch.div(composition_weight, structure_weight+composition_weight)
        return composition_weight_final, structure_weight_final

    def forward(self, composition_input, structure_input, structure_presence):
        composition_representation = self.composition_network(composition_input)
        structure_representation = self.structure_network(structure_input)

        composition_weight, structure_weight = self.weight_determination(composition_representation,
                                                                         structure_representation,
                                                                         structure_presence)
        if self.element_sum:
            combined_representation = composition_representation*composition_weight + structure_representation*structure_weight
        else:
            combined_representation = torch.cat((composition_representation*composition_weight, structure_representation*structure_weight),1)
        combined_representation = self.combined_representation_update_network(combined_representation)
        pred = self.output_network(combined_representation)
        pred = self.non_negative_1(pred) # make band gap non-negative
        return pred
