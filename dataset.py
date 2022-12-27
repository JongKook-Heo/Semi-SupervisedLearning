import numpy as np
from copy import deepcopy
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import Data as PyG_Data
from torch_geometric.data import Dataset

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from auglichem.utils import (
        ATOM_LIST,
        CHIRALITY_LIST,
        BOND_LIST,
        BONDDIR_LIST,
        random_split,
        scaffold_split
)
from auglichem.molecule import RandomAtomMask, RandomBondDelete, MotifRemoval, Compose
from auglichem.molecule.data._load_sets import read_smiles


#TODO docstrings for MoleculeData

class DualMoleculeDataset(Dataset):
    def __init__(self, dataset, data_path=None, s_transform=None, w_transform=None, smiles_data=None, labels=None,
                 task=None, test_mode=False, aug_time=0, atom_mask_ratio=[0, 0.25],
                 bond_delete_ratio=[0, 0.25], target=None, class_labels=None, seed=None,
                 augment_original=False, _training_set=False, _train_warn=True, **kwargs):
        '''
            Initialize Molecular Data set object. This object tracks data, labels,
            task, test, and augmentation

            Input:
            -----------------------------------
            dataset (str): Name of data set
            data_path (str, optional default=None): path to store or load data
            smiles_data (np.ndarray of str): Data set in smiles format
            labels (np.ndarray of float): Data labels, maybe optional if unsupervised?
            task (str): 'regression' or 'classification' indicating the learning task
            test_mode (boolean, default=False): Does no augmentations if true
            aug_time (int, optional, default=1): Controls augmentations (not quite sure how yet)
            atom_mask_ratio (list, float): If list, sample mask ratio uniformly over [a, b]
                                           where a < b, if float, set ratio to input.
            bond_delete_ratio (list, float): If list, sample mask ratio uniformly over [a, b]
                                           where a < b, if float, set ratio to input.


            Output:
            -----------------------------------
            None
        '''
        super(Dataset, self).__init__()

        # Store class attributes
        self.dataset = dataset
        self.data_path = data_path

        # Handle incorrectly passed in transformations
        if(isinstance(s_transform, list)):
            s_transform = Compose(s_transform)
        elif(not(isinstance(s_transform, Compose))):
            s_transform = Compose([s_transform])
        self.s_transform = s_transform
        
        if(isinstance(w_transform, list)):
            w_transform = Compose(w_transform)
        elif(not(isinstance(w_transform, Compose))):
            w_transform = Compose([w_transform])
        self.w_transform = w_transform
        
        if(smiles_data is None):
            self.smiles_data, self.labels, self.task = read_smiles(dataset, data_path)
        else:
            self.smiles_data = smiles_data
            self.labels = labels
            self.task = task

        # No augmentation if no transform is specified
        if(self.w_transform is None):
            self.test_mode = True
        else:
            self.test_mode = test_mode
            self.aug_time = aug_time

        if self.test_mode:
            self.aug_time = 0

        assert type(self.aug_time) == int

        # For reproducibility
        self.seed = seed
        if(seed is not None):
            np.random.seed(self.seed)
        self.reproduce_seeds = list(range(self.__len__()))
        np.random.shuffle(self.reproduce_seeds)

        # Store mask ratios
        self.atom_mask_ratio = atom_mask_ratio
        self.bond_delete_ratio = bond_delete_ratio

        if(target is not None):
            self.target = target
        else:
            self.target = list(self.labels.keys())[0]

        if(class_labels is not None):
            self.class_labels = class_labels
        else:
            self.class_labels = self.labels[self.target]

        # Augment original data
        self.augment_original = augment_original
        if(self.augment_original):
            warnings.warn("Augmenting original dataset may lead to unexpected results.",
                          RuntimeWarning, stacklevel=2)

        self._training_set = _training_set
        self._train_warn = _train_warn
        self._handle_motifs_s()
        self._handle_motifs_w()

    def _handle_motifs_s(self):
        # MotifRemoval adds multiple new SMILES strings to our data, and must be done
        # upon training set initialization
        if(self._training_set):
            if(self._train_warn): # Catches if not set through get_data_loaders()
                raise ValueError(
                    "_training_set is for internal use only. " + \
                    "Manually setting _training_set=True is not supported."
                )

            s_transform = []
            for t in self.s_transform.transforms:
                if(isinstance(t, MotifRemoval)):
                    print("Gathering Motifs...")
                    new_smiles = []
                    new_labels = []

                    # Match label to each motif
                    for smiles, labels in tqdm(zip(self.smiles_data, self.class_labels)):
                        aug_mols = t(smiles)
                        new_smiles.append(smiles)
                        new_labels.append(labels)
                        for am in aug_mols:
                            new_smiles.append(Chem.MolToSmiles(am))
                            new_labels.append(labels)

                    self.smiles_data = np.array(new_smiles)
                    self.class_labels = np.array(new_labels)
                else:
                    s_transform.append(t)
            self.s_transform = Compose(s_transform)
        else:
            pass

    def _handle_motifs_w(self):
        # MotifRemoval adds multiple new SMILES strings to our data, and must be done
        # upon training set initialization
        if(self._training_set):
            if(self._train_warn): # Catches if not set through get_data_loaders()
                raise ValueError(
                    "_training_set is for internal use only. " + \
                    "Manually setting _training_set=True is not supported."
                )

            w_transform = []
            for t in self.w_transform.transforms:
                if(isinstance(t, MotifRemoval)):
                    print("Gathering Motifs...")
                    new_smiles = []
                    new_labels = []

                    # Match label to each motif
                    for smiles, labels in tqdm(zip(self.smiles_data, self.class_labels)):
                        aug_mols = t(smiles)
                        new_smiles.append(smiles)
                        new_labels.append(labels)
                        for am in aug_mols:
                            new_smiles.append(Chem.MolToSmiles(am))
                            new_labels.append(labels)

                    self.smiles_data = np.array(new_smiles)
                    self.class_labels = np.array(new_labels)
                else:
                    w_transform.append(t)
            self.w_transform = Compose(w_transform)
        else:
            pass
        
    def _get_data_x(self, mol):
        '''
            Get the transformed data features.

            Inputs:
            -----------------------------------
            mol ( object): Current molecule

            Outputs:
            -----------------------------------
            x (torch.Tensor of longs):
        '''

        # Set up data arrays
        type_idx, chirality_idx, atomic_number = [], [], []

        # Gather atom data
        for atom in mol.GetAtoms():
            try:
                if ATOM_LIST.index(atom.GetAtomicNum()) == 119:
                    print(atom.GetAtomicNum())
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                atomic_number.append(atom.GetAtomicNum())
            except ValueError: # Skip asterisk in motif
                pass

        # Concatenate atom type with chirality index
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        return x


    def _get_data_y(self, index):
        '''
            Get the transformed data label.

            Inputs:
            -----------------------------------
            index (int): Index for current molecule

            Outputs:
            -----------------------------------
            y (torch.Tensor, long if classification, float if regression): Data label
        '''

        if self.task == 'classification':
            y = torch.tensor(self.class_labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.class_labels[index], dtype=torch.float).view(1,-1)

        return y


    def _get_edge_index_and_attr(self, mol):
        '''
            Create the edge index and attributes

            Inputs:
            -----------------------------------
            mol ():

            Outputs:
            -----------------------------------
            edge_index ():
            edge_attr ():
        '''

        # Set up data collection lists
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():

            # Get the beginning and end atom indices
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Store bond atoms
            row += [start, end]
            col += [end, start]

            # Store edge featuers
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        # Create edge index and attributes
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        return edge_index, edge_attr


    def __getitem__(self, index):
        '''
            Selects an element of self.smiles_data according to the index.
            Edge and node masking are done here for each individual molecule

            Input:
            -----------------------------------
            index (int): Index of molecule we would like to augment

            Output:
            -----------------------------------
            masked_data (Data object): data that has been augmented with node and edge masking

        '''
        # If augmentation is done, actual dataset is smaller than given indices
        if self.test_mode:
            true_index = index
        else:
            true_index = index // (self.aug_time + 1)

        # Create initial data set
        mol = Chem.MolFromSmiles(self.smiles_data[true_index])
        mol = Chem.AddHs(mol)

        # Get data x and y
        x = self._get_data_x(mol)
        y = self._get_data_y(true_index)

        # Get edge index and attributes
        edge_index, edge_attr = self._get_edge_index_and_attr(mol)

        # Set up PyG data object
        molecule = PyG_Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                            smiles=self.smiles_data[true_index])
        
        aug_molecule_s = self.s_transform(molecule, seed=self.reproduce_seeds[index])
        aug_molecule_w = self.w_transform(molecule, seed=self.reproduce_seeds[index])

        return aug_molecule_w, aug_molecule_s    

    def _clean_label(self, target):
        """
            Removes labels that don't have values for a given target.
        """
        # If value is not -999999999 we have a valid mol - label pair
        good_idxs = []
        for i, val in enumerate(self.labels[target]):
            if(val != -999999999):
                good_idxs.append(i)
        return good_idxs


    def _match_indices(self, good_idx):
        """
            Match indices between our train/valid/test index and the good indices we have
            for each data subset.
        """

        # For each of train, valid, test, we match where we have a valid molecular
        # representation with where we have a valid label
        updated_train_idx = []
        for v in self.train_idx:
            try:
                updated_train_idx.append(good_idx.index(v))
            except ValueError:
                # No label for training data
                pass

        updated_valid_idx = []
        for v in self.valid_idx:
            try:
                updated_valid_idx.append(good_idx.index(v))
            except ValueError:
                # No label for training data
                pass

        updated_test_idx = []
        for v in self.test_idx:
            try:
                updated_test_idx.append(good_idx.index(v))
            except ValueError:
                # No label for training data
                pass

        # Update indices
        self.train_idx = updated_train_idx
        self.valid_idx = updated_valid_idx
        self.test_idx = updated_test_idx


    def __len__(self):
        return len(self.smiles_data) * (self.aug_time + 1) # Original + augmented
