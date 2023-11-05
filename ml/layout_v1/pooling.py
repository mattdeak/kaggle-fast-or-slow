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
        degree: torch.Tensor,
        alpha: int = 0,
    ) -> torch.Tensor:
        r"""
        Args:
            edge_index (LongTensor): graph connectivity in COO format, shape (2, E)
        Returns:
            Tensor: output scalar, shape (1,)
        """
        s = (torch.log(degree + 1) / self.avg_degree) ** alpha
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


def multi_agg(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return torch.hstack(
        [
            global_mean_pool(x, batch),
            global_std_pool(x, batch),
            global_max_pool(x, batch),
            global_min_pool(x, batch),
        ]
    )


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
        """
        Args:
            x (Tensor): input feature, shape (N, C, *)
            edge_index (LongTensor): graph connectivity in COO format, shape (2, E)
            batch (LongTensor): batch vector, shape (N,)
        Returns:
            Tensor: output feature, shape (B, O) where O = C * len(aggregators) * (scales + 1)
        """
        ...
        # The following code works for a single graph, but not for a batch of graphs.
        num_nodes = x.shape[0]
        # in_degree = degree(edge_index[1, :], num_nodes)
        
        
        index[1, :]),
        
        
        

        degrees = torch.stack(
            [
                self.scaler(in_degree, alpha=1),
                self.scaler(in_degree, alpha=0.5),
            ]
        ).T

        x_unsqueeze = x.unsqueeze(1)
        degrees_unsqueeze = degrees.unsqueeze(2)

        result = (x_unsqueeze * degrees_unsqueeze).view(num_nodes, -1)
        # add back the original features
        result = torch.hstack([x, result])

        aggs = torch.hstack([agg(result, batch) for agg in self.aggregators])

        # compute tensor product of features and degrees
        return aggs

    def __repr__(self):
        return f"{self.__class__.__name__}(avg_degree={self.avg_degree}, aggregators={self.aggregators})"
