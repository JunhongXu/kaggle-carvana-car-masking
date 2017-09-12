# kaggle-carvana-car-masking
Kaggle Competition, hope I can win this one!

8.29: added a weighted bce loss, do not use vertical flip, do not shift the image too much. boundary weighting is calculated by average pooling, interesting idea!

9.2: RefineNet-v2 gets a better result. Validation accuracy at 0.9974 and LB: 0.9965

9.12: RefineNet-V4 with hierarchical training: 1024 * 1024 -> 1152 * 1152 -> 1280 * 1792. Training: 0.00567/0.9980; Validation: 0.00691/0.9978

-----------

RefineNet-v1: Pre-trained ResNet-50 with features added to decoder
RefineNet-v2: Pre-trained ResNet-50 with features stacked to decoder
RefineNet-v3: Train with image dimension of (640*960)  Not so good: 0.9961

Add more augmentation: Just use horizontal flip and rotation.

------------

TODO:

1. Try Dense CRF for post-processing
