# Google - Fast or Slow (Top 3%)
This repo contains the code used to get the 14th place (top 3%) solution. Since it's kaggle and therefor a bit ad-hoc, some of it is a little messy. The bulk of the relevant code is in the `ml/layout_v1` folder.

## Problem Definition
A detailed explanation of the competition can be found [here](https://www.kaggle.com/competitions/predict-ai-model-runtime/overview). In summary, we need to rank configurations for different neural networks to optimize runtime. 

## Layout Solution
### Graph Reduction
Each layout graph was transformed into two distinct graphs:

1. A 3-Hop graph (hops from the Configurable Nodes)
    I think this is likely a common strategy - the configurable nodes are the ones that can differ between graphs, so it makes sense that any graph reduction would try to preserve these nodes. An N-Hop graph transformation will retain configurable nodes, and nodes (and edges) up to N hops away from any configurable node. I landed on a 3-hop graph through val scores, but I think we could have gotten better results with 4 or 5 hop graphs given some time to tune.
2. A "Config Positioning " graph (name pending). This graph removed all non-configurable nodes, but drew edges between all configurable nodes where there was a path from one node to another that did not cross another configurable node. My intuition guiding this was that the relative position of a poorly configured node with respect to downstream configurable nodes might have a meaningful impact on its overall contribution to the runtime. 

A small example of the transformations is shown below (though I just used a 1-hop example because I don't want to draw all night).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2F3ecc17ccdeead41cba76a26a6087d03b%2FGraphReductions3.png?generation=1700281118388593&alt=media)

### The Features
Before even talking about new features, it's worth mentioning that normalization is absolutely essential on this problem. If you didn't normalize your data, you didn't score well. Ok, onwards.

#### Node Features
We defined a few extra features. For the nodes, we defined:
* Shape sparsity (shape sum / shape product)
* Dimensionality (count of active shapes)
* Stride Interactions
* Padding Proportions
* Reversal Ratio
* Is configurable (obvious)

I'm not actually convinced any of these features led to meaningful lift in scores, but they were present in the final model.

#### Configuration Features
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

#### Opcodes
Just one-hot encoded. I tried a couple other things, like grouping them by intuitive similarity, but I wasn't able to come up with a grouping that reliably improved performance.

#### Global Features
* Longest Path Length in Graph
* Average shortest path length between connected components
* Number of nodes
* is_default

The is_default flag was introduced because it _seemed_ like the random vs. default distributions were different enough to warrant having predictive value, since the test set also contained this information. I was initially worried that this would hurt generalization, but it seemed to provide a small but reliable boost to val scores.


### The Models
We defined the following GraphBlock, using a GAT with (out channels / 2) channels to process the node features given the 3-Hop Graph, and GAT or GraphSAGE with (out channels / 2) channels to process the features given the Config Positioning graph.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2F893f9ce0ec19a542d2156afc8e951c66%2FGraphBlock2.png?generation=1700278642575220&alt=media)

We then layered the GraphBlocks with residual connections and added some dense feedforward layers (also with residuals) to which we concatenated the global features. The final result was an ensemble of slightly different versions of this model (varying hidden dims, linear layers, etc.). 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3626485%2Fc9cfbe2d9197c006df390155fbaab177%2FModelDiagram.png?generation=1700279330883081&alt=media)


#### Training Process
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

#### Ensembling
For a given file id (e.g "xla:default:abc..."), we have N (1000 or 1001) predictions per model output. We min-max normalize the predictions so they're all in a zero-to-one range, then we just add the scores elementwise for each model output. We use the summed config scores to derive the rank ordering.

The normalization is important here, because the models are not guaranteed to be outputting numbers on the same scale if you're using a ranking loss.

I also tried simple rank averaging and Borda count, both of which worked somewhat but not as well as the min-max averaging. I'd bet this is because these methods can't account for things like "how much better is rank 2 than rank 3", while the min-max normalized ensemble can.

## Tile
I won't go into as much depth here because it's not very interesting. I just processed the XLA Tile graphs (max 1000 configs per graph) and used a similar model as above, except that I just used GAT w/ 3 graph layers and 3 feedforward layers. I can go into details if requested, but honestly it scored a 0.197/0.2 on the public leaderboard early on and I never went back to it.
