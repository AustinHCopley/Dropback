# Dropback

Running this requires PyTorch and torchvision.

I used PyTorch with CUDA version 2.0.1+cu117
& torchvision version 0.15.2+cu117

The data used is from MNIST and CIFAR10. These are not included in the submission, but they are optionally downloaded when running the jupyter notebooks

Training and tests are done in jupyter notebooks because it makes it much easier for me personally to make changes and debug while keeping the runtime and without losing defined variables.

But the results from every test are stored in a dictionary which is saved as a pickle file: model_results.pkl, where the dictionary is keyed by a tuple, and the values are also tuples:

```python 
dict[(freeze_epoch: int, prune_threshold: float)] = (train_accuracies: list, validation_accuracies: list, model: DropBackMLP)
```

The baseline data is stored in the baselin_mnist.pkl file, and loading this will just return the tuple:

```python
(train_accuracies: list, validation_accuracies: list, model: DropBackMLP)
```

Training and testing should work with both CPU and GPU, however I did not extensively test CPU functionality, so there could possibly be issues with data being on both devices if not run with CUDA

Clearer details on the experiments/tests are in the [dropback notebook](dropback.ipynb)

Visualization and results graphs are found in the [vis notebook](vis.ipynb)