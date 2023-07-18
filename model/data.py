import os
from pickle import NONE
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from model.composition import generate_features, _element_composition
import json
from tqdm import tqdm
import math
import random

random.seed(1)


from pymatgen.core.structure import Structure
from pymatgen.core import Composition, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from sklearn.metrics import mean_absolute_error

import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

crystal_system_to_int = {
        "triclinic": 1,
        "monoclinic": 2,
        "orthorhombic": 3,
        "tetragonal": 4,
        "trigonal": 5,
        "hexagonal": 6,
        "cubic": 7
        }

def convert_composition(composition,elem_features,device):
    #here, roost is used to prepare inputs of structures

    comp_dict = Composition(composition).get_el_amt_dict()
    elements = list(comp_dict)
    weights = list(comp_dict.values())
    weights = np.atleast_2d(weights).T / np.sum(weights)
    try:
        elem_fea = np.vstack([elem_features[element] for element in elements])
    except AssertionError:
        raise AssertionError(
                f"({composition}) contains element types not in embedding"
            )
    except ValueError:
        raise ValueError(
                f"({composition}) composition cannot be parsed into elements"
            )
    
    nele = len(elements)
    self_idx = []
    nbr_idx = []
    for i, _ in enumerate(elements):
        self_idx += [i] * nele
        nbr_idx += list(range(nele))

    # convert all data to tensors
    elem_weights = Tensor(weights).to(device)
    elem_fea = Tensor(elem_fea).to(device)
    self_idx = LongTensor(self_idx).to(device)
    nbr_idx = LongTensor(nbr_idx).to(device)

    return (elem_weights, elem_fea, self_idx, nbr_idx)


def convert_structure(structure,elem_features,rcut,max_num_nbr,device):
    #here, de-cgcnn is used to prepare inputs of structures
    if structure is None:
        #create a fake structure as a place holder
        lattice = Lattice.cubic(1)
        structure = Structure(lattice, ['H'], [[0,0,0]])
    
    def GaussianDistance(distances, dmin=0, dmax=rcut, step=0.2):
        filter = np.arange(dmin, dmax+step, step)
        var = step
        return np.exp(-(distances[..., np.newaxis] - filter)**2 / var**2)

    def get_atom_fea(structure):
        atom_feas = []
        for atom in structure:
            for i, specie in enumerate(atom.species):
                if i == 0:
                    fea = np.array(elem_features[specie.symbol])*atom.species.get_atomic_fraction(specie)
                else:
                    fea = fea + np.array(elem_features[specie.symbol])*atom.species.get_atomic_fraction(specie)
            atom_feas.append(fea)
        atom_feas = np.vstack(atom_feas)
        return Tensor(atom_feas).to(device)  

    def get_cell_fea(structure):
        cell_fea = []
        structure = structure.get_primitive_structure(use_site_props=True)
        lattice_constants = structure.lattice.abc
        for d in lattice_constants:
            cell_fea.append(d)
        lattice_angles = structure.lattice.angles
        for d in lattice_angles:
            cell_fea.append(d)
        sg_analyzer = SpacegroupAnalyzer(structure)
        crystal_system = sg_analyzer.get_crystal_system()
        crystal_system_int = crystal_system_to_int[crystal_system]
        space_group_number = sg_analyzer.get_space_group_number()
        cell_fea.append(crystal_system_int)
        cell_fea.append(space_group_number)
        return Tensor([cell_fea]).to(device)

    atom_fea = get_atom_fea(structure)

    all_nbrs = structure.get_all_neighbors(rcut, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +[0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [rcut + 1.] * (max_num_nbr -len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = GaussianDistance(nbr_fea)
    nbr_fea = Tensor(nbr_fea).to(device)
    nbr_fea_idx = LongTensor(nbr_fea_idx).to(device)

    cell_fea = get_cell_fea(structure)

    return (atom_fea, nbr_fea, nbr_fea_idx, cell_fea)


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_cell_fea, crystal_atom_idx = [], [], [], [], []
    batch_elem_weights, batch_elem_fea, batch_self_idx, batch_nbr_idx, crystal_elem_idx = [], [], [], [], []
    batch_target = []; batch_ids = []
    batch_structure_presence = []
    number_of_element, number_of_atom = 0, 0
    for i, (id, (composition_input, structure_input, structure_presence), target) \
         in enumerate(dataset_list):
        elem_weights, elem_fea, self_idx, nbr_idx = composition_input
        atom_fea, nbr_fea, nbr_fea_idx, cell_fea = structure_input
        natom = atom_fea.shape[0] # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+number_of_atom)
        new_idx = torch.LongTensor(np.arange(natom)+number_of_atom)
        crystal_atom_idx.append(new_idx)
        batch_cell_fea.append(cell_fea)
        number_of_atom += natom

        n_ele = elem_fea.shape[0] # number of elements for this crystal
        batch_elem_fea.append(elem_fea)
        batch_elem_weights.append(elem_weights)
        batch_self_idx.append(self_idx + number_of_element)
        batch_nbr_idx.append(nbr_idx + number_of_element)
        crystal_elem_idx.append(torch.tensor([i]*n_ele))
        number_of_element += n_ele

        batch_target.append(target)
        batch_structure_presence.append(structure_presence)
        batch_ids.append(id)
    
    return batch_ids, \
           (torch.cat(batch_elem_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_elem_idx)), \
           (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_cell_fea, dim=0),
            crystal_atom_idx), \
           (torch.stack(batch_target, dim=0),
            torch.stack(batch_structure_presence, dim=0))
        
#basic data loader to load the csv file and the cif files
class MutimodalData(Dataset):
    def __init__(self,
              datapath,
              device,
              csvpath,
              structurepath,
              std = 1.,
              train=False,
              rcut=8.0,
              max_num_nbr = 12,
              elem_embedding: str = "matscholar200"):

        self.std = std

        self.device = device

        csvpath = os.path.join(datapath, csvpath)
 #       structurepath = os.path.join(datapath, structurepath)
        df = pd.read_csv(csvpath)

        with open(os.path.join('utils/element',elem_embedding+'.json')) as file:
             elem_features = json.load(file)
    
        self.elem_emb_len = len(elem_features[list(elem_features.keys())[0]])
        self.data = []
        props = []
        n_structure = 0

        if train:
            comp_temp_icsd_index = {}
        
        for index, row in tqdm(list(df.iterrows())):
            single_data = dict()
            composition = df.at[index, 'composition']
            composition = composition.replace(' ','')
            target = float(df.at[index, 'target']) 
            props.append(target)

            icsd = df.at[index, 'mp-id']
            if not icsd: #no icsd number, no structure
                icsd = None
                structure = None
                structure_presence = 0
            else:
                if os.path.isfile(os.path.join(structurepath,str(icsd))):
                    structure = Structure.from_file(os.path.join(structurepath,str(icsd)))
                    structure_presence = 1
                    n_structure += 1
                    if train:
                        identifier = str(composition)
                        if identifier not in list(comp_temp_icsd_index.keys()):
                            comp_temp_icsd_index[identifier] = [index]
                        else:
                            comp_temp_icsd_index[identifier].append(index)
                else:
                    icsd = None
                    structure = None
                    structure_presence = 0

            single_data['id'] = composition+'_'+str(icsd)
            single_data['composition_input'] = convert_composition(composition,elem_features,device)
            single_data['structure_input'] = convert_structure(structure,elem_features,rcut,max_num_nbr,device)
            single_data['structure_presence'] = structure_presence
            single_data['icsd'] = icsd
            single_data['target'] = target

            self.data.append(single_data)

        print ('%d compostions, %d structures loaded'%(index+1, n_structure))
        
        if train:
            self.std = np.std(props)
            n_augmented = 0
            #augment training set: for those composition+temperature that only have one structure, augment by removing the structure
            for identifier in comp_temp_icsd_index.keys():
                if len(comp_temp_icsd_index[identifier]) > 1:
                    continue
                n_augmented += 1
                index = comp_temp_icsd_index[identifier][0]
                single_data = dict()
                composition = df.at[index, 'composition']
                composition = composition.replace(' ','')
                target = float(df.at[index, 'target']) 

                icsd = None
                structure = None
                structure_presence = 0

                single_data['id'] = composition+'_'+str(icsd)
                single_data['composition_input'] = convert_composition(composition,elem_features,device)
                single_data['structure_input'] = convert_structure(structure,elem_features,rcut,max_num_nbr,device)
                single_data['structure_presence'] = structure_presence
                single_data['icsd'] = icsd
                single_data['target'] = target

                self.data.append(single_data)

            random.shuffle(self.data)
            print ('%d augmented data added'%(n_augmented))
                   
        for i in range(len(self.data)):
            self.data[i]['target'] = self.data[i]['target'] / self.std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        id = self.data[index]['id']
        composition_input = self.data[index]['composition_input']
        structure_input = self.data[index]['structure_input']
        structure_presence = Tensor([self.data[index]['structure_presence']]).to(self.device)
        target = Tensor([self.data[index]['target']]).to(self.device)

        return id, (composition_input, structure_input, structure_presence), target



