import torch
from torch import nn

from wild_relation_network.layers import GroupObjectsIntoPairs, DeepLinearBNReLU, Identity, GroupObjectsIntoTriples


class RelationNetwork(nn.Module):
    """
    Relation Network (RN) [1].
    The originally proposed model works on object pairs.
    This implementation allows to extend the RN to work on object triples (by setting tuple_size=3).
    Using larger tuples may be impractical, as the memory requirement grows exponentially,
    with complexity O(num_objects ^ tuple_size).

    [1] Santoro, Adam, et al. "A simple neural network module for relational reasoning." NeurIPS 2017
    """

    def __init__(self, num_objects: int, object_size: int, out_size: int, g_depth: int = 3,
                 f_depth: int = 2, use_object_triples: bool = False, use_layer_norm: bool = False):
        """
        Initializes the RN model.
        :param num_objects: number of objects (objects are represented as 1D tensors)
        :param object_size: object size
        :param out_size: output size
        :param g_depth: number of MLP layers in G
        :param f_depth: number of MLP layers in F
        :param use_object_triples: flag indicating whether to group objects into triples (by default object pairs are considered)
        :param use_layer_norm: flag indicating whether layer normalization should be applied after G submodule
        """
        super(RelationNetwork, self).__init__()
        self.group_objects = GroupObjectsIntoTriples(num_objects) if use_object_triples else GroupObjectsIntoPairs(num_objects)
        object_tuple_size = (3 if use_object_triples else 2) * object_size
        self.g = G(depth=g_depth, in_size=object_tuple_size, out_size=object_tuple_size, use_layer_norm=use_layer_norm)
        self.f = F(depth=f_depth, object_size=object_tuple_size, out_size=out_size, dropout_probability=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RN model.
        :param x: a tensor with shape (batch_size, num_objects, object_size)
        :return: a tensor with shape (batch_size, out_size)
        """
        x = self.group_objects(x)
        x = self.g(x)
        x = self.f(x)
        return x


class G(nn.Module):
    def __init__(self, depth: int, in_size: int, out_size: int, use_layer_norm: bool = False):
        super(G, self).__init__()
        self.mlp = DeepLinearBNReLU(depth, in_size, out_size, change_dim_first=False)
        self.norm = nn.LayerNorm(out_size) if use_layer_norm else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.sum(dim=1)
        x = self.norm(x)
        return x


class F(nn.Module):
    def __init__(self, depth: int, object_size: int, out_size: int, dropout_probability: float = 0.5):
        super(F, self).__init__()
        self.mlp = nn.Sequential(
            DeepLinearBNReLU(depth, object_size, object_size),
            nn.Dropout(dropout_probability),
            nn.Linear(object_size, out_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
