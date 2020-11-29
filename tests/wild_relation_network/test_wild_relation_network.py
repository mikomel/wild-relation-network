import pytest
import torch
import torch.nn.functional as F
import torchtest

from wild_relation_network import WReN


@pytest.mark.parametrize('use_object_triples', [False, True])
@pytest.mark.parametrize('use_layer_norm', [False, True])
def test_forward(use_object_triples, use_layer_norm):
    x = torch.rand(4, 16, 160, 160)
    y = torch.randint(8, (4,), dtype=torch.long)
    wren = WReN(use_object_triples=use_object_triples, use_layer_norm=use_layer_norm)
    optimiser = torch.optim.Adam(wren.parameters())
    torchtest.test_suite(
        model=wren,
        loss_fn=F.cross_entropy,
        optim=optimiser,
        batch=[x, y],
        test_inf_vals=True,
        test_nan_vals=True,
        test_vars_change=True
    )
