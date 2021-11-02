# Cifar100 Result

# Overall

- Cifar100 dataset is loaded using tf.keras.datasets module, with 50000 images for training and 10000 images for testing.
- Split 50000 images for training (above) into 40000 for actual training and 10000, which is used to create validation set
- The first 2 models (mentioned below) use Adam optimizer with learning rate 1e-3, the last one uses Adam optimizer with learning rate 1e-4.
- Preprocessing images by scaling to 0 - 1 range.
- Loss: Categorical crossentropy.
- Metric: Accuracy.
- Number of epochs: 40

## Result

- Custom model with 2 CNNs and 1 FC has ~90% accuracy on the training set but only ~50% accuracy on the validation set and test set. It is clear that the model has overfitted the training set.
- ResNet50 performs badly on Cifar100, has only roughly 40% accuracy on the validation set and test set but 98% on the training set. Heavily overfitted the training set (see screenshot attached in screenshots folder).
- Custom model with 3 CNNs and 1 FC: this model uses ELU activation (except for the last before softmax layer). ELU activation eliminates dead neurons issue by shifting bias mean toward zero, meanwhile RELU activation has positive bias mean, and value smaller than zero become zero make corresponding neuron dead, since RELU does not have gradient at zero. With 3 CNNs (Conv2D -> Activation -> Conv2D -> BN -> Activation -> Pooling -> Dropout) and 1 FC (Dense -> Activation -> Softmax), the model has around 53.3% accuracy on the validation set and test set and ~75% accuracy on the training set (overfitted).
- Attempted to insert BN before Activation as well as expand the training set by data augmentation but it did not improve much.
