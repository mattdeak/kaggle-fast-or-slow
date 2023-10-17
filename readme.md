## Notes so far
* XGBoost w summing node features does pretty good (approx 0.191/0.2 on Tile Dataset)
* GraphSage with large-ish channel space w/ sum aggregations current best tile score at 0.197.
  * Weight Decay (1e-3), no dropout. (dropout didn't hurt too much, but weight decay alone seemed better)
  * Fairly deep (6 graph blocks, 3 linear layers) with residual connections. Layernorm absolutely required, otherwise results were not stable.
  * Log-normalizing input seemed to help, maybe by reducing the magnitude of gradients out from the loss?
  * Global Add Pooling

* Other types of graph convolutions that were OK but definitely worse. 
  * GCNConv and GAT

Still haven't tried:
* pairwise loss functions. Notebooks seem to suggest good results with pairwise loss, but
  the fact that I regressed directly on the config runtime makes me suspicious that there were maybe
  architectural problems when other people tried to regress (e.g missing layer norms, extreme regression targets, or something).
  still worth trying though.
* More complicated global pooling. Add pooling made intuitive sense to me, but we can definitely
  get fancier and use global attention pooling or something that makes more use of graph structure.
* haven't even started working on layout dataset yet. It's much bigger, which probably comes with some
  of it's own challenges.



