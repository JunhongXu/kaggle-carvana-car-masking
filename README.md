# kaggle-carvana-car-masking
Kaggle Competition, hope I can win this one!

8.29: added a weighted bce loss, do not use vertical flip, do not shift the image too much. boundary weighting is calculated by average pooling, interesting idea!

9.2: RefineNet-v2 gets a better result. Validation accuracy at 0.9974 and LB: 0.9965

-----------

RefineNet-v1: Pre-trained ResNet-50 with features added to decoder
RefineNet-v2: Pre-trained ResNet-50 with features stacked to decoder

-----------

In progress: 

1. Train with image dimension of (800*1200)
2. Add more augmentation

------------

TODO:

1. Try DenseNet.
2. Try a larger batch size.
3. Make predictions on different scales and average scores.
