import torch
from typing import Optional


class GaussianDistance(torch.nn.Module):
    """
    Expands the distance by Gaussian basis.

    Creates a Gaussian basis set and projects the scalar interatomic distance
    onto this basis set to form a continuous, fixed-length feature vector.
    """

    def __init__(
        self,
        dmin: float = 0.0,
        dmax: float = 8.0,
        step: float = 0.2,
        var: Optional[float] = None,
    ):
        """
        Args:
            dmin (float): Minimum distance.
            dmax (float): Maximum distance.
            step (float): Step size for the Gaussian filter.
            var (float, optional): Variance of the Gaussian filter.
                                   Defaults to step if not provided.
        """
        super().__init__()
        if dmin >= dmax:
            raise ValueError(f"dmin ({dmin}) must be less than dmax ({dmax})")
        if step <= 0:
            raise ValueError(f"step ({step}) must be greater than 0")

        self.filter = torch.arange(dmin, dmax + step, step, dtype=torch.float32)
        if var is None:
            var = step
        self.var = var

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances using the Gaussian basis.

        Args:
            distances (torch.Tensor): Tensor of shape (N,) or (N, 1)
                                      containing interatomic distances.

        Returns:
            torch.Tensor: Tensor of shape (N, len(self.filter)) containing
                          the expanded distances.
        """
        distances = distances.view(-1, 1)
        # Move filter to the same device as distances
        filter_tensor = self.filter.to(distances.device)
        return torch.exp(-((distances - filter_tensor) ** 2) / self.var**2)
