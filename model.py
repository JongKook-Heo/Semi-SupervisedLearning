import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from auglichem.utils import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG, NUM_BOND_TYPE, NUM_BOND_DIRECTION


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        # print(edge_index)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINE(nn.Module):
    def __init__(self, 
                 task='classification',
                 emb_dim=300, 
                 feat_dim=256,
                 num_layer=5,
                 pool='mean',
                 drop_ratio=.0,
                 output_dim=None,
                 **kwargs):
        
        super(GINE, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task
        if(output_dim == None):
            self.output_dim = 1
        else:
            self.output_dim = output_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE + 1, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == 'classification':
            self.output_dim *= 2
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.output_dim)
            )
        elif self.task == 'regression':
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.feat_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.feat_dim//4),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//4, self.output_dim)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.feat_lin(h)
        h = self.pool(h, data.batch)

        return h, self.pred_lin(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)
            
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def __call__(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param + self.decay * self.shadow[
                        name
                    ]
                    self.shadow[name] = new_average

    def assign(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.clone()
                param.data.copy_(self.shadow[name].data)

    def resume(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name].data)