import torch
from typing import Optional
from torch_geometric.nn import SGConv

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(
    edge_index, edge_weight=None, fill_value=1, num_nodes=None
):
    # reposition the diagonal values to the end
    """
    edge_weight.shape : (num_nodes*num_nodes*batch_size,)
    ()
    """
    # actually return num_nodes
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col

    inv_mask = ~mask  # diagonal positions
    # print("inv_mask", inv_mask)

    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device,
    )

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]

        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight

        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(NewSGConv, self).__init__(
            num_features, num_classes, K=K, cached=cached, bias=bias
        )
        torch.nn.init.xavier_normal_(self.lin.weight)
        self.norm_ = None

    # allow negative edge weights
    @staticmethod
    # Note: here,num_nodes=self.num_nodes*batch_size
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )
        row, col = edge_index

        deg = scatter_add(
            torch.abs(edge_weight), row, dim=0, dim_size=num_nodes
        )  # calculate degreematrix, i.e, D(stretched) in the paper.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        # calculate normalized adjacency matrix, i.e, S(stretched) in the paper.
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, self.norm_ = NewSGConv.norm(
                edge_index,
                x.size(0),
                edge_weight,
                dtype=x.dtype,
            )

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=self.norm_)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j):
        return self.norm_.view(-1, 1) * x_j


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)
