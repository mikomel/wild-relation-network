import pytest
import torch

from wild_relation_network import RelationNetwork


@pytest.mark.parametrize('use_object_triples', [False, True])
@pytest.mark.parametrize('use_layer_norm', [False, True])
def test_forward(use_object_triples, use_layer_norm):
    x = torch.rand(4, 8, 64)
    rn = RelationNetwork(
        num_objects=8,
        object_size=64,
        out_size=32,
        use_object_triples=use_object_triples,
        use_layer_norm=use_layer_norm
    )
    logits = rn(x)
    assert logits.shape == (4, 32)
