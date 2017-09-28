# kaggle-carvana-car-masking



## Progress

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
  
9.17: Pesudo labeling not performing so well.

9.18: Pre-trained classification not performing well.

9.26: Score on private LB: 0.9969/ Ranking: 29



## Models

1. RefineNet-v1: Pre-trained ResNet-50 with features added to decoder
2. RefineNet-v2: Pre-trained ResNet-50 with features stacked to decoder
3. Refinenet-v3: Pre-trained ResNet-50 with features stacked to decoder
4. RefineNet-v4: Pre-trained ResNet-34 with stacked features. (Trained on 1024->1152->(1280*1792))
5. RefineNet-v5: Pre-trained VGG-16 with stacked features.
6. RefineNet-v6: Pre-trained VGG-16 with auxiliary loss.


## Final ensembles

1. refinenetv4_resnet34_1280*1920
2. refinenetv4_resnet34_1280*1536
3. refinenetv4_resnet34_1280*1280
4. refinenetv2_resnet50_1024*1024
5. refinenetv3_resnet50_1024*1024
6. refinenetv5_vgg_1024
7. refinenetv6_1024
8. refinenetv1_1024

## Lessons

1. Cross validation (good train/val splits) gives better results.
2. Network structure matters. Seems stacking the features instead of adding them gives slightly better results.
3. Using pre-trained network helps convergence.
4. Curriculum learning makes learning speed faster (train from low res to high res.).

Did not use CV, all the networks were trained using same hyper-parameters and train/val split. This should be something prevents me from getting into top 20.
In addition, synthetic data did increase the accuracy, but instead making larger models overfitting.



