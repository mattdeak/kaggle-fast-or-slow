# Context
* Business Context: https://www.kaggle.com/competitions/predict-ai-model-runtime/overview
* Data Context: https://www.kaggle.com/competitions/predict-ai-model-runtime/data

# Overview of Approach
## Data Preprocessing
* Developed 2 graph compression techniques to reduce problem complexity (N-hop reduction from config nodes and a config meta-graph).
* Normalized numeric features, dropped features with 0 standard deviation
* One-hot encoded opcodes

## Feature Engineering
* Created several node specific features and config specific features
* Created some global features applied to the whole graph

## Model Design
The models all broadly followed the following format:
1. Graph/Config/Opcodes concatenated
2. Graph representations used to perform some Graph Convolutions (varies slightly between models)
3. Global Mean Pooling concatenated with Global Features
4. MLP to output layer


The Tile Dataset result was a single model following this design, with 3 GraphSAGE layers and 3 Linear Layers trained with ListMLE loss. The Layout Dataset results were taken from an ensemble of models with slight variations in their design. All models used GeLu activations, but differed in other respects (detailed below). Output losses used were ListMLE and Pairwise Hinge.

## Validation
We kept the same Train/Val split as provided in the competition dataset.

# Details of Approach 
## Graph Reduction
Each layout graph was transformed into two distinct graphs:

1. A 3-Hop graph (hops from the Configurable Nodes)
    The configurable nodes are the ones that can differ between graphs, so it makes sense that any graph reduction would try to preserve these nodes. An N-Hop graph transformation will retain configurable nodes, and nodes (and edges) up to N hops away from any configurable node. I landed on a 3-hop graph through val scores, but I think we could have gotten better results with 4 or 5 hop graphs given some time to tune.
2. A "Config Positioning " graph. This graph removed all non-configurable nodes, but drew edges between all configurable nodes where there was a path from one node to another that did not cross another configurable node. My intuition guiding this was that the relative position of a poorly configured node with respect to downstream configurable nodes might have a meaningful impact on its overall contribution to the runtime. 
An example of the transformations is shown below (though I just used a 1-hop example).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2F3ecc17ccdeead41cba76a26a6087d03b%2FGraphReductions3.png?generation=1700281118388593&alt=media)

## The Features
Before even talking about new features, it's worth mentioning that normalization is absolutely essential on this problem. If you didn't normalize your data, you didn't score well.

### Node Features
We defined a few extra features. For the nodes, we defined:
* Shape sparsity (shape sum / shape product)
* Dimensionality (count of active shapes)
* Stride Interactions
* Padding Proportions
* Reversal Ratio
* Is configurable (obvious)

I'm not actually convinced any of these features led to meaningful lift in scores, but they were present in the final model.

### Configuration Features
Additional config features were computed for the output, input and kernel sections. Each feature was replicated for each of those sections:
* is_default (all negative ones)
* active_dims (count of non-negative)
* max order (largest value)
* contiguity rank (count of longest contiguous ordering / active dims)
* section variance

Additionally, we computed similarity metrics for
* output-input
* output-kernel
* input-kernel

### Opcodes
Just one-hot encoded. I tried a couple other things, like grouping them by intuitive similarity, but I wasn't able to come up with a grouping that reliably improved performance.

### Global Features
* Longest Path Length in Graph
* Average shortest path length between connected components
* Number of nodes
* is_default

The is_default flag was introduced because it _seemed_ like the random vs. default distributions were different enough to warrant having predictive value, since the test set also contained this information. I was initially worried that this would hurt generalization, but it seemed to provide a small but reliable boost to val scores.

## The Models
#### Layout Models
For the layout problem, we defined the following GraphBlock, using a GAT with (out channels / 2) channels to process the node features given the 3-Hop Graph, and GAT or GraphSAGE with (out channels / 2) channels to process the features given the Config Positioning graph.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2F893f9ce0ec19a542d2156afc8e951c66%2FGraphBlock2.png?generation=1700278642575220&alt=media)

We then layered the GraphBlocks with residual connections and added some dense feed-forward layers (also with residuals) to which we concatenated the global features. The final result was an ensemble of slightly different versions of this model (varying hidden dims, linear layers, etc.). 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2Fc9cfbe2d9197c006df390155fbaab177%2FModelDiagram.png?generation=1700279330883081&alt=media)

The code for the pytorch module used in the Layout Set is shown below (the Tile Set model is very similar, with fewer complicated pieces. It's less polished but you can view it [here](https://github.com/mattdeak/kaggle-fast-or-slow/blob/master/ml/xla_gcn_v1/model.py).


```python
class SAGEBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        with_residual: bool = True,
        dropout: float = 0.5,
        graph_norm: Literal["graph", "layer"] = "graph",
    ):
        super().__init__()
        self.conv = SAGEConv(input_dim, output_dim)
        self.norm_type = graph_norm
        self.norm = build_norm(graph_norm, output_dim)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

        self.output_dim = output_dim

    def forward(self, d: Data) -> Data:
        x, edge_index, batch = d.x, d.edge_index, d.batch

        f = self.conv(x, edge_index)
        f = F.gelu(f)

        if self.norm_type == "graph":
            f = self.norm(f, batch)
        else:
            f = self.norm(f)

        f = self.dropout(f)

        if self.with_residual:
            f += x

        new_data = Data(x=f, edge_index=edge_index, batch=batch)
        return d.update(new_data)


class GATBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 4,
        with_residual: bool = True,
        dropout: float = 0.5,
        graph_norm: Literal["graph", "layer"] = "graph",
    ):
        super().__init__()

        gat_output_dim = output_dim // heads

        self.conv = GATConv(input_dim, gat_output_dim, heads=heads)
        self.norm_type = graph_norm
        self.norm = build_norm(graph_norm, output_dim)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

        self.output_dim = output_dim

    def forward(self, d: Data) -> Data:
        x, edge_index, batch = d.x, d.edge_index, d.batch

        f = self.conv(x, edge_index)
        f = F.gelu(f)

        if self.norm_type == "graph":
            f = self.norm(f, batch)
        else:
            f = self.norm(f)

        f = self.dropout(f)

        if self.with_residual:
            f += x

        new_data = Data(x=f, edge_index=edge_index, batch=batch)
        return d.update(new_data)


class MultiEdgeGATBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        heads: int = 4,
        with_residual: bool = True,
        dropout: float = 0.5,
        main_block: Literal["gat", "sage"] = "gat",
        alt_block: Literal["gat", "sage"] = "sage",
    ):
        """A block that applies two different edge convolutions to the graph, and then
        concatenates the results together. Uses an edge mask to determine which edges
        to apply the main block to, and which to apply the alternate block to.
        """

        super().__init__()

        output_dim_per_block = output_dim // 2

        if main_block == "gat":
            self.main_edge_block = GATBlock(
                input_dim,
                output_dim_per_block,
                heads=heads,
                with_residual=False,
                dropout=dropout,
            )
        else:
            self.main_edge_block = SAGEBlock(
                input_dim,
                output_dim_per_block,
                with_residual=False,
                dropout=dropout,
            )

        if alt_block == "gat":
            self.alternate_edge_block = GATBlock(
                input_dim,
                output_dim_per_block,
                heads=heads,
                with_residual=False,
                dropout=dropout,
            )

        else:
            self.alternate_edge_block = SAGEBlock(
                input_dim,
                output_dim_per_block,
                with_residual=False,
                dropout=dropout,
            )

        self.with_residual = with_residual
        self.output_dim = output_dim

    def forward(self, data: Data):
        main_edge_index = data.edge_index
        alternate_edge_index = data.alt_edge_index

        main_edge_data = Data(
            x=data.x,
            edge_index=main_edge_index,
            batch=data.batch,
        )

        alternate_edge_data = Data(
            x=data.x,
            edge_index=alternate_edge_index,
            batch=data.batch,
        )

        main_edge_data = self.main_edge_block(main_edge_data)
        alternate_edge_data = self.alternate_edge_block(alternate_edge_data)

        f = torch.cat([main_edge_data.x, alternate_edge_data.x], dim=1)

        if self.with_residual:
            f += data.x

        new_data = Data(
            x=f,
            batch=data.batch,
        )

        return data.update(new_data)

class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        with_residual: bool = True,
        dropout: float = 0.5,
        linear_norm: Literal["layer", "batch"] = "layer",
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = build_norm(linear_norm, output_dim)
        self.with_residual = with_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.linear(x)
        f = F.gelu(f)
        f = self.norm(f)
        f = self.dropout(f)

        if self.with_residual:
            f += x

        return f


class GraphMLP(nn.Module):
    def __init__(
        self,
        graph_input_dim: int = INPUT_DIM,
        global_features_dim: int | None = GLOBAL_INPUT_DIM,
        graph_channels: int = 64,
        graph_layers: int = 6,
        linear_channels: int = 32,
        linear_layers: int = 3,
        dropout: float = 0.2,
        pooling_fn: GlobalPoolingFn = global_mean_pool,
        pooling_feature_multiplier: int = 1,
        graph_conv: Literal["sage", "gat"] = "sage",
        graph_conv_kwargs: dict[str, Any] | None = None,
        use_multi_edge: bool = True,
        graph_norm: Literal["graph", "layer"] = "graph",
        linear_norm: Literal["layer", "batch"] = "layer",
        main_block: Literal["gat", "sage"] = "gat",
        alt_block: Literal["gat", "sage"] = "sage",
    ):
        super().__init__()

        self.use_multi_edge = use_multi_edge
        self.pooling_fn = pooling_fn
        self.gcns = nn.ModuleList()

        build_conv = self._create_conv_factory(
            graph_channels,
            graph_conv,
            graph_conv_kwargs,
            dropout,
            use_multi_edge,
            graph_norm,
            main_block,
            alt_block,
        )

        block = build_conv(
            input_dim=graph_input_dim,
            with_residual=False,  # because the first dim is different
        )
        self.gcns.append(block)

        for _ in range(graph_layers):
            block = build_conv(
                input_dim=block.output_dim,
                with_residual=True,
            )
            self.gcns.append(block)

        if global_features_dim:
            first_linear_input_dim = (
                pooling_feature_multiplier * block.output_dim + global_features_dim
            )
        else:
            first_linear_input_dim = pooling_feature_multiplier * block.output_dim

        self.mlp = nn.Sequential(
            LinearBlock(
                first_linear_input_dim,
                linear_channels,
                with_residual=False,
                dropout=dropout,
                linear_norm=linear_norm,
            ),
            *[
                LinearBlock(
                    linear_channels,
                    linear_channels,
                    dropout=dropout,
                    linear_norm=linear_norm,
                )
                for _ in range(linear_layers)
            ],
            nn.Linear(linear_channels, 1),
        )

    def _create_conv_factory(
        self,
        graph_channels: int,
        graph_conv: Literal["sage", "gat"],
        graph_conv_kwargs: dict[str, Any] | None,
        dropout: float,
        use_multi_edge: bool,
        graph_norm: Literal["graph", "layer"],
        main_block: Literal["gat", "sage"],
        alt_block: Literal["gat", "sage"],
    ) -> ConvFactory:
        def build_conv(
            input_dim: int, with_residual: bool
        ) -> MultiEdgeGATBlock | SAGEBlock | GATBlock:
            if use_multi_edge:
                return MultiEdgeGATBlock(
                    input_dim=input_dim,
                    output_dim=graph_channels,
                    dropout=dropout,
                    with_residual=with_residual,
                    main_block=main_block,
                    alt_block=alt_block,
                    **(graph_conv_kwargs or {}),
                )

            if graph_conv == "sage":
                return SAGEBlock(
                    input_dim,
                    graph_channels,
                    dropout=dropout,
                    with_residual=with_residual,
                    graph_norm=graph_norm,
                )
            else:
                return GATBlock(
                    input_dim,
                    graph_channels,
                    dropout=dropout,
                    with_residual=with_residual,
                    graph_norm=graph_norm,
                    **(graph_conv_kwargs or {}),
                )

        return build_conv

    def forward(self, data: Data) -> torch.Tensor:
        d = data
        for gcn_block in self.gcns:
            d = gcn_block(d)

        pool = self.pooling_fn(d.x, d.batch)  # type: ignore

        if data.global_features is not None:
            # shape we need from global features is (batch, global_features_dim)
            # shape we have is (batch * global_features_dim)
            pool = torch.cat([pool, data.global_features], dim=1)

        return self.mlp(pool)

    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        d = data
        for gcn_block in self.gcns:
            d = gcn_block(d)

        pool = self.pooling_fn(d.x, d.batch)  # type: ignore
        return pool
```

#### Tile Model
The Model for the Tile Dataset is more or less exactly the same, except it used 3 graph layers, 3 linear layers, and no graph reduction at all (because there were no configurable node).

### Training Process
NLP and XLA were trained separately. I would have loved to play with a unified model more, but I ran out of time and compute, and they seemed to learn well when they were separate.

All models used:
* AdamW Optimizer
* Batch Size of 16
* GeLu Activations
* LayerNorm in both Graph and MLP
* Global Mean Pooling after the graph blocks
* No Scheduler

Other parameters are as follows:

| Parameter         | XLA-1        | XLA-2        | NLP-1        | NLP-2            |
|-------------------|--------------|--------------|--------------|------------------|
| Loss              | listMLE      | listMLE      | listMLE      | Rank Margin Loss |
| Learning Rate     | 0.00028      | 0.00028      | 0.00028      | 0.0001           |
| Weight Decay      | 0.004        | 0.004        | 0.004        | 0.007            |
| Graph Layers      | 4            | 4            | 4            | 4                |
| Graph Channels    | 128          | 128          | 128          | 128              |
| FF Layers         | 1            | 2            | 1            | 3                |
| FF Channels       | 128          | 128          | 128          | 64               |
| 3-Hop Graph Conv  | GAT(heads=8) | GAT(heads=8) | GAT(heads=8) | GAT(heads=1)     |
| Config Graph Conv | GAT          | GAT          | GAT          | GraphSAGE        |
| Dropout           | 0.15         | 0.15         | 0            | 0                |

Outputs were collected from XLA-1 after 3 epochs, and 2 snapshots of XLA-2 in training (end of epochs 2 and 3) based on their val scores.

The NLP Models never even finished one epoch, the loss appeared to plateau and I only had so much compute. This indicates to me that I probably could have tuned the LR better or regularized better (although increasing regularization seemed to hurt rather than help in my sweeps).

### Ensembling
For a given file id (e.g "xla:default:abc..."), we have N (1000 or 1001) predictions per model output. We min-max normalize the predictions so they're all in a zero-to-one range, then we just add the scores elementwise for each model output. We use the summed config scores to derive the rank ordering.

The normalization is important here, because the models are not guaranteed to be outputting numbers on the same scale if you're using a ranking loss.

I also tried simple rank averaging and Borda count, both of which worked but not as well as the min-max averaging. This is likely because these methods can't account for things like "how much better is rank 2 than rank 3", while the min-max normalized ensemble can.

#### Sources
* Graph Attention: https://arxiv.org/pdf/1710.10903.pdf
* GraphSAGE: https://arxiv.org/pdf/1706.02216.pdf
