import numpy as np
from olorenchemengine.internal import mock_imports

try:
    import torch
    import torch_geometric.data
except:
    mock_imports(globals(), "torch", "torch_geometric")

try:
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
except ImportError:
    mock_imports(globals(), "atom_to_feature_vector", "bond_to_feature_vector")

from rdkit import Chem


def molecule_to_graph(mol, include_mol=False):
    # Convert to RDKit molecule
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    # Generate nodes of the graph
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # Generate edges of the graph
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    if include_mol:
        graph["mol"] = mol

    return graph


def molecule_to_pyg(smiles_str, y=None):
    graph = molecule_to_graph(smiles_str)  # construct ogb graph
    g = torch_geometric.data.Data()
    g.__num_nodes__ = graph["num_nodes"]
    g.edge_index = torch.from_numpy(graph["edge_index"])

    del graph["num_nodes"]
    del graph["edge_index"]

    if graph["edge_feat"] is not None:
        g.edge_attr = torch.from_numpy(graph["edge_feat"])
        del graph["edge_feat"]

    if graph["node_feat"] is not None:
        g.x = torch.from_numpy(graph["node_feat"])
        del graph["node_feat"]

    if y is not None:
        if type(y) == bool:
            g.y = torch.LongTensor([1 if y else 0])
        else:
            g.y = torch.FloatTensor([y])

    return g


def batch_molecule_to_pyg(smiles_list, y=None):
    graphs = [molecule_to_pyg(smiles, y) for smiles in smiles_list]
    return torch_geometric.data.Batch.from_data_list(graphs)
