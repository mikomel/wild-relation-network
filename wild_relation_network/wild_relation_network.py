import torch
from torch import nn

from wild_relation_network import relation_network
from wild_relation_network.layers import ConvBNReLU, GroupObjectsIntoPairs, GroupObjectsIntoPairsWith, TagPanelEmbeddings, Identity, \
    GroupObjectsIntoTriples, GroupObjectsIntoTriplesWith


class WReN(nn.Module):
    """
    Wild Relation Network (WReN) [1] for solving Raven's Progressive Matrices.
    The originally proposed model uses a Relation Network (RN) [2] which works on object pairs.
    This implementation allows to extend the RN to work on object triples (by setting use_object_triples=True).
    Using larger tuples is impractical, as the memory requirement grows exponentially,
    with complexity O(num_objects ^ rn_tuple_size).
    After extension to triples, the model resembles the Logic Embedding Network (LEN) [3].

    [1] Santoro, Adam, et al. "Measuring abstract reasoning in neural networks." ICML 2018
    [2] Santoro, Adam, et al. "A simple neural network module for relational reasoning." NeurIPS 2017
    [3] Zheng, Kecheng, Zheng-Jun Zha, and Wei Wei. "Abstract reasoning with distracting features." NeurIPS 2019
    """

    def __init__(self, num_channels: int = 32, use_object_triples: bool = False, use_layer_norm: bool = False):
        """
        Initializes the WReN model.
        :param num_channels: number of convolutional kernels in each CNN layer
        :param use_object_triples: flag indicating whether to group objects into triples (by default object pairs are considered)
        in the Relation Network submodule. Use False to reproduce WReN and True to reproduce LEN.
        :param use_layer_norm: flag indicating whether layer normalization should be applied after
        the G submodule of RN.
        """
        super(WReN, self).__init__()
        if use_object_triples:
            self.group_objects = GroupObjectsIntoTriples(num_objects=8)
            self.group_objects_with = GroupObjectsIntoTriplesWith()
        else:
            self.group_objects = GroupObjectsIntoPairs(num_objects=8)
            self.group_objects_with = GroupObjectsIntoPairsWith()

        self.cnn = nn.Sequential(
            ConvBNReLU(1, num_channels, kernel_size=3, stride=2),
            ConvBNReLU(num_channels, num_channels, kernel_size=3, stride=2),
            ConvBNReLU(num_channels, num_channels, kernel_size=3, stride=2),
            ConvBNReLU(num_channels, num_channels, kernel_size=3, stride=2)
        )
        self.object_size = num_channels * 9 * 9
        self.object_tuple_size = (3 if use_object_triples else 2) * (self.object_size + 9)
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.g = relation_network.G(
            depth=3,
            in_size=self.object_tuple_size,
            out_size=self.object_tuple_size,
            use_layer_norm=False
        )
        self.norm = nn.LayerNorm(self.object_tuple_size) if use_layer_norm else Identity()
        self.f = relation_network.F(
            depth=2,
            object_size=self.object_tuple_size,
            out_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WReN model.
        :param x: a tensor with shape (batch_size, num_panels, height, width). num_panels is assumed
        to be 16, as the model was designed to solve RPMs from the PGM dataset.
        :return: a tensor with shape (batch_size, num_answers). num_answers is always equal to 8,
        which is the number of answers for each RPM in PGM.
        """
        batch_size, num_panels, height, width = x.size()
        x = x.view(batch_size * num_panels, 1, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, self.object_size)
        x = self.tag_panel_embeddings(x)
        context_objects = x[:, :8, :]
        choice_objects = x[:, 8:, :]
        context_pairs = self.group_objects(context_objects)
        context_g_out = self.g(context_pairs)
        f_out = torch.zeros(batch_size, 8, device=x.device).type_as(x)
        for i in range(8):
            context_choice_pairs = self.group_objects_with(context_objects, choice_objects[:, i, :])
            context_choice_g_out = self.g(context_choice_pairs)
            relations = context_g_out + context_choice_g_out
            relations = self.norm(relations)
            f_out[:, i] = self.f(relations).squeeze()
        return f_out
