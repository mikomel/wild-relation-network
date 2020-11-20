import pytest
import torch

from wild_relation_network import WReN


@pytest.mark.parametrize('use_object_triples', [False, True])
@pytest.mark.parametrize('use_layer_norm', [False, True])
def test_forward(use_object_triples, use_layer_norm):
    x = torch.rand(4, 16, 160, 160)
    wren = WReN(use_object_triples=use_object_triples, use_layer_norm=use_layer_norm)
    logits = wren(x)
    assert logits.shape == (4, 8)
