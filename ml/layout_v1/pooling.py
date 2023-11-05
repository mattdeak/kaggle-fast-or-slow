import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torch_geometric.utils import degree


class DegreeScaler(nn.Module):
    def __init__(self, avg_log_degree: float = 1.0):
        """
        The avg log degree is 1/|train| * sum_{i in train}(log(d+1)) where d is the average degree of the graph.
        """
        super().__init__()
        self.avg_degree = avg_log_degree

    def forward(
        self,
        edge_index: torch.Tensor,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        r"""
        Args:
            edge_index (LongTensor): graph connectivity in COO format, shape (2, E)
        Returns:
            Tensor: output scalar, shape (1,)
        """
        in_degree = degree(edge_index[1])
        s = (torch.log(in_degree + 1) / self.avg_degree) ** alpha
        return s


def global_std_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    r"""
    Args:
        x (Tensor): input feature, shape (N, C, *)
        batch (LongTensor): batch vector, shape (N,)
    Returns:
        Tensor: output feature, shape (B, C, *)
    """
    return torch_scatter.scatter_std(x, batch, dim=0)


def global_min_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return global_max_pool(-x, batch)


class DegreeScaledGlobalPooler(nn.Module):
    AGGREGATORS = {
        "mean": global_mean_pool,
        "std": global_std_pool,
        "max": global_max_pool,
        "min": global_min_pool,
    }

    def __init__(
        self,
        avg_degree: float = 1.0,
        aggregators: list[str] = ["mean", "std", "max", "min"],
    ):
        """
        Args:
            avg_degree (float): average degree of the graph. Must be precomputed.
        """
        super().__init__()
        self.avg_degree = avg_degree
        self.aggregators = [self.AGGREGATORS[agg] for agg in aggregators]
        self.scaler = DegreeScaler(avg_log_degree=avg_degree)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        degrees = torch.hstack(
            [
                self.scaler(edge_index, 0),
                self.scaler(edge_index, 1),
                self.scaler(edge_index, -1),
            ]
        )
        agg_features = [agg(x, batch) for agg in self.aggregators]

        # compute tensor product of features and degrees
        return torch.cat([agg * degrees for agg in agg_features], dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(avg_degree={self.avg_degree}, aggregators={self.aggregators})"
