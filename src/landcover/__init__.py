"""Land cover classification of Sentinel-2 imagery with a transfer learned ResNet-50.

The package holds the whole pipeline for the project: dataset indexing and
splitting, the transfer learning model, training, evaluation and figures.

Modules
-------
config      Paths, dataset constants and hyperparameters.
data        Dataset discovery, the 80/10/10 split and reproducibility records.
dataset     Torch datasets and the image transforms.
model       ResNet-50 with a replaced classification head.
train       The two stage training loop.
evaluate    Test set scoring, classification report and confusion matrix.
predict     Single image inference from a saved checkpoint.
plot        Figures for the README and the documentation.
metrics     Metric helpers shared by evaluation, training, figures and tests.
smoke       A minute long end to end run on synthetic patches.
"""

__version__ = "0.1.0"
