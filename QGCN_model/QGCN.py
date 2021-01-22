"""
QGCN model implementation. First there are layers of GCN, then a bilinear layer (the last layer of the QGCN), and then
there is the QGCN which have them both.
"""

import json
from torch.nn import Module, Linear, Dropout, Embedding, ModuleList, Sequential, LogSoftmax
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

ADAM_ = Adam
SGD_ = SGD
relu_ = torch.relu
sigmoid_ = torch.sigmoid
tanh_ = torch.tanh


"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""


# class of GCN model
class GCN(Module):
    def __init__(self, in_dim, out_dim, activation, dropout):
        super(GCN, self).__init__()
        # useful info in forward function
        self._linear = Linear(in_dim, out_dim)
        self._activation = activation
        self._dropout = Dropout(p=dropout)
        self._gpu = False

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0)
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.matmul(A, x0)

        x = torch.log(torch.abs(self._linear(Ax)) + 1e-4)
        x1 = self._activation(x)
        return x1

    @ staticmethod
    def return_ax(A, x0):
        return torch.matmul(A, x0)


# class of last layer of the QGCN
class QGCNLastLayer(Module):
    def __init__(self, left_in_dim, right_in_dim, out_dim):
        super(QGCNLastLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(left_in_dim, 1)
        self._right_linear = Linear(right_in_dim, out_dim)
        self._gpu = False

    def forward(self, A, x0, x1):
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.matmul(x1.permute(0, 2, 1), A)
        W2_x1_A = self._left_linear(x1_A.permute(0, 2, 1))
        W2_x1_A_x0 = torch.matmul(W2_x1_A.permute(0, 2, 1), x0)
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        return W2_x1_A_x0_W3.squeeze(dim=1)


# final model of QGCN
class QGCN(Module):
    """
    first linear layer is executed numerous times
    """""

    def __init__(self, params, in_dim, embed_vocab_dim):
        super(QGCN, self).__init__()
        self._params = params["model"] if type(params) is dict else json.load(open(params, "rt"))["model"]

        # add embedding layers if needed, also is the data binary or not
        self._is_binary = True if self._params["label_type"] == "binary" else False
        self._is_embed = True if self._params["use_embeddings"] == "True" else False

        # dimensions of the GCN layers according to the params json file
        qgcn_layers_dim = [{"in": layer["in_dim"], "out": layer["out_dim"]} for layer in self._params["GCN_layers"]]
        qgcn_layers_dim[0]["in"] = in_dim

        # if embedding is given there are also layer for the embedding
        if self._is_embed:
            # embeddings are added to ftr vector -> update dimensions of relevant weights
            qgcn_layers_dim[0]["in"] += sum(self._params["embeddings_dim"])

            # add embedding layers
            self._embed_layers = []
            for vocab_dim, embed_dim in zip(embed_vocab_dim, self._params["embeddings_dim"]):
                self._embed_layers.append(Embedding(vocab_dim, embed_dim))
            self._embed_layers = ModuleList(self._embed_layers)

        self._num_layers = len(self._params["GCN_layers"])
        self._linear_layers = []
        # create linear layers
        self._linear_layers = ModuleList([GCN(qgcn_layers_dim[i]["in"], qgcn_layers_dim[i]["out"],
                                              globals()[self._params["activation"]], self._params["dropout"])
                                          for i in range(self._num_layers)])
        qgcn_right_in = in_dim + sum(self._params["embeddings_dim"])
        self._qgcn_last_layer = QGCNLastLayer(left_in_dim=qgcn_layers_dim[-1]["out"], right_in_dim=qgcn_right_in,
                                              out_dim=1 if self._is_binary else self._params["num_classes"])
        self._softmax = LogSoftmax(dim=1)
        self.optimizer = self.set_optimizer(self._params["lr"], globals()[self._params["optimizer"]], self._params["L2_regularization"])

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _fix_shape(self, input):
        return input if len(input.shape) == 3 else input.unsqueeze(dim=0)

    def forward(self, A, x0, embed):
        if self._is_embed:
            list_embed = []
            for i, embedding in enumerate(self._embed_layers):
                list_embed.append(embedding(embed[:, :, i]))
            x0 = torch.cat([x0] + list_embed, dim=2)

        x1 = x0
        for i in range(self._num_layers):
            x1 = self._linear_layers[i](A, x1)

        x2 = self._qgcn_last_layer(A, x0, x1)
        x2 = torch.sigmoid(x2) if self._is_binary else self._softmax(x2)
        return x2
