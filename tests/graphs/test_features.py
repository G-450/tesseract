import pytest
import torch

from src.graphs.features import GaussianDistance


def test_gaussian_distance_initialization() -> None:
    gd = GaussianDistance(dmin=0.0, dmax=8.0, step=0.2)
    assert gd.filter.shape[0] == 41  # (8.0 - 0.0) / 0.2 + 1
    assert gd.filter[0].item() == 0.0
    assert abs(gd.filter[-1].item() - 8.0) < 1e-5
    assert gd.var == 0.2


def test_gaussian_distance_invalid_initialization() -> None:
    with pytest.raises(ValueError):
        GaussianDistance(dmin=8.0, dmax=0.0, step=0.2)

    with pytest.raises(ValueError):
        GaussianDistance(dmin=0.0, dmax=8.0, step=-0.2)


def test_gaussian_distance_forward() -> None:
    gd = GaussianDistance(dmin=0.0, dmax=2.0, step=1.0)
    # filter = [0.0, 1.0, 2.0], var = 1.0
    distances = torch.tensor([0.0, 1.0, 2.0])

    output = gd(distances)

    assert output.shape == (3, 3)

    # Distance 0.0
    assert torch.allclose(
        output[0, :],
        torch.tensor(
            [1.0, torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(-4.0))]
        ),
    )

    # Distance 1.0
    assert torch.allclose(
        output[1, :],
        torch.tensor(
            [torch.exp(torch.tensor(-1.0)), 1.0, torch.exp(torch.tensor(-1.0))]
        ),
    )

    # Distance 2.0
    assert torch.allclose(
        output[2, :],
        torch.tensor(
            [torch.exp(torch.tensor(-4.0)), torch.exp(torch.tensor(-1.0)), 1.0]
        ),
    )
