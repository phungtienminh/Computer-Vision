# Cifar100 Result

# Version 3
- v3.1: Slightly increase in dropout rates reduces overfitting a little bit, the accuracy increases to 68.25%.
- v3.0: The accuracy on the test set is 67% after making some improvement in data augmentation.

# Version 2

- Standardization is used instead of normalization.
- With gradient clipping and data augmentation, the accuracy on the test set is 64%.
- No. epochs: 100

# Version 1

# Overall

- Cifar100 dataset is loaded using tf.keras.datasets module, with 50000 images for training and 10000 images for testing.
- Split 50000 images for training (above) into 40000 for actual training and 10000, which is used to create validation set
- The first 2 models (mentioned below) use Adam optimizer with learning rate 1e-3, the last one uses Adam optimizer with learning rate 1e-4.
- Preprocessing images by scaling to 0 - 1 range.
- Loss: Categorical crossentropy.
- Metric: Accuracy.
- Number of epochs: 40

## Result

- 53.3% accuracy on the test set.
