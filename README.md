# kaggle-carvana-car-masking
Kaggle Competition, hope I can win this one!

8.29: added a weighted bce loss, do not use vertical flip, do not shift the image too much. boundary weighting is calculated by average pooling, interesting idea!

9.2: RefineNet-v2 gets a better result. Validation accuracy at 0.9974 and LB: 0.9965

9.12: RefineNet-V4 with hierarchical training: 1024 * 1024 -> 1152 * 1152 -> 1280 * 1792. Training: 0.00567/0.9980; Validation: 0.00691/0.9978

9.13: Ensemble-V1 gets the best result so far: 0.9969/LB:17. 

    ensembles = [
                'refinenetv4_resnet34_1280*1280_hq',    # 0.9967, augmented by 2
                'refinenetv4_resnet34_1024*1024_hq',    # 0.9966
                'refinenetv4_1024_hq',                  # 0.9965
                'refinenetv3_resnet50_1024*1024_hq'     # 0.9965
            ]

-----------

RefineNet-v1: Pre-trained ResNet-50 with features added to decoder
RefineNet-v2: Pre-trained ResNet-50 with features stacked to decoder
RefineNet-v4: Pre-trained ResNet-34 with stacked features. (Trained on 1024->1152->(1280*1792))

------------

TODO:

1. Try more ensembles and look into difference between these single models.
2. Try add a classification layer (this should be done in the dataset class).
3. Try training 1280*1920 on RefineNetV4 (This is the simplest one to implement).
