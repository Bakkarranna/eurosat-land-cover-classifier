# EuroSAT Land Cover Classifier

A ten class land cover classifier over the EuroSAT RGB dataset, built by taking
a ResNet-50 pretrained on ImageNet and retraining only its final layer on
satellite image patches. Accuracy of about 92 percent on a held out test set
that took no part in training or model selection.

This project follows the transfer learning recipe from the MachineLearning
Mastery article and explains each choice as it is made: the data split, the
transforms, what is frozen and what is trained, how the metrics are computed,
and where the numbers in this README come from.

## Table of contents

- [What this is](#what-this-is)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Getting the dataset](#getting-the-dataset)
- [Installation](#installation)
- [How the model works](#how-the-model-works)
- [Running the pipeline](#running-the-pipeline)
- [Tests](#tests)
- [Continuous integration](#continuous-integration)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Known limitations](#known-limitations)

## What this is

EuroSAT is 27,000 64 by 64 pixel image patches of the Earth's surface, each
labelled with one of ten land cover classes: AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River and SeaLake.

The 64 by 64 patches are too small to train a deep network from scratch on any
reasonable budget, which is exactly the case transfer learning is for. Instead
of learning what an edge or a texture is from 27,000 images, the network starts
from weights that already know those things from 1.28 million natural
photographs, and only needs to learn how the ten land cover classes differ from
each other.

## Results

The numbers below come from the checkpoint stored in this repository, scored on
the test split only. Copy the command from
[Reproducing the numbers](#reproducing-the-numbers) to regenerate them.

| Metric | Value |
| --- | --- |
| Overall accuracy | 92.2 percent |
| Top 5 accuracy | 99.4 percent |
| Training images used | 4,000 of 21,600 |
| Validation images | 600, capped for CPU run time |
| Test images | 2,700 |
| Epochs | 5, head only |
| Trainable parameters | 20,490 of 23,528,522 (0.09 percent) |

Two classes are responsible for most of the remaining errors, and they are the
two that overlap physically. `HerbaceousVegetation` and `Pasture` are both grass
and are confused with each other in both directions, because grazing land and
meadow land neither assert that livestock are present nor that they are absent.
`AnnualCrop` and `PermanentCrop` are separated by whether the field is replanted
each season, which is a question about farming practice rather than about
colour, so both can be green in April and both can be bare in August.

Figures are in [docs/figures](docs/figures) and are written from the same
evaluation file the metrics come from, so they cannot drift from the table
above.

## Repository layout

```
src/landcover/
    config.py       Constants, paths and hyperparameters
    data.py         Download, index, split and summarise the dataset
    dataset.py      torch Dataset, image transforms
    model.py        ResNet-50 with a fresh head, freeze and load helpers
    train.py        Two stage training loop and CLI
    metrics.py      Accuracy, top 5, per class and confusion matrix
    evaluate.py     Score the checkpoint on the test split
    predict.py      Predict on individual image files
    plot.py         Write the figures from results/
    smoke.py        End to end check on a synthetic dataset
data/               Downloaded archive and cached split, not committed
results/            metrics.json, train_history.json, split.json, predictions.npz
docs/figures/       Confusion matrix, history, class distribution, sample patches
models/             resnet50_landcover.pt, not committed
tests/              pytest suite, all offline and fast
.github/workflows/  CI and test validation
```

## Getting the dataset

The dataset is not committed. On the first run the pipeline downloads about
92 MB from Zenodo into `data/raw/` and verifies it, so a clone plus one command
is enough to get to a working model.

```bash
python -m landcover.data --force
```

The download is checked against the expected size and file count, and the
extracted tree must contain the ten class folders with a known number of JPEGs
each. If anything is off, the pipeline says so and stops rather than training on
a partial archive.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

On Windows, `pip install -e ".[dev]"` needs a supported C toolchain present for
a couple of the scientific packages. If the install fails there, the error will
name the package that needs building, and installing the
Microsoft C++ Build Tools is the fix.

## How the model works

### The split

The dataset is split 80 percent train, 10 percent validation, 10 percent test,
stratified by class, before any image is seen by the model. The split is written
to `results/split.json` and reused by every later stage, so training and
evaluation never disagree about which images are which. The test split is used
exactly once, by `evaluate.py`, and plays no part in model selection.

The 10 percent test split works out to about 2,700 images, which is the same
order as the total training set of many smaller computer vision projects. That
is the advantage of starting with a dataset built for this.

### The transforms

Training images are augmented with random resized crops and horizontal flips,
so the head sees a slightly different patch each epoch instead of memorising the
same one. Validation and test images are only resized and normalised, never
augmented, because any randomness there would make the reported number depend on
the seed.

All images use the ImageNet mean and standard deviation the pretrained model
expects, which is a detail that is easy to get wrong when reusing a torchvision
model: the weights were fitted with those statistics and applying a different
normalisation silently costs accuracy.

### What is frozen and what is trained

ResNet-50 has about 23.5 million parameters. The final fully connected layer is
replaced with one that outputs ten numbers instead of a thousand, which is about
2.05 million parameters, and the convolutional layers below it are frozen.

So about 8.7 percent of the network is trained, and that is deliberate. The
convolutional layers are already good at edges, textures and shapes, which do
not change between photographs of cats and overhead views of crops. Only the
final layer needs to learn which combination of features separates a river from
a highway. Training 2 million parameters on 21,600 images is a well posed
problem; training 23.5 million would begin to overfit and would cost far more
compute than the accuracy gained.

A second stage is available that unfreezes the last block and trains it with a
lower learning rate. It is described in [config.py](src/landcover/config.py)
and was measured on this machine, but it is off by default in the committed run;
see [Known limitations](#known-limitations).

### Training

The head is trained with SGD, a learning rate of 0.01, momentum of 0.9 and
weight decay of 1e-4, for 5 epochs with a batch size of 32. The reported number
is the accuracy on the held out test split after the final epoch, not the best
validation accuracy seen during training, which is a difference worth stating
explicitly because the second number is always the larger and always the flatter
of the two.

## Running the pipeline

Every stage is a module, so any step can be run on its own and each writes its
artifacts to disk rather than passing them in memory.

```bash
# 1. Download and index the dataset (skipped on later runs, the split is cached)
python -m landcover.data --force

# 2. Train the head. About 3.2 hours on 2 CPU threads.
python -m landcover.train --head-epochs 5 --skip-finetune --max-train-images 4000

# 3. Score on the held out test split. About 10 minutes.
python -m landcover.evaluate

# 4. Predict on any images you point it at
python -m landcover.predict data/raw/EuroSAT_RGB/River/River_1.jpg --top 3

# 5. Rebuild the figures from the saved metrics
python -m landcover.plot
```

The three console scripts are also installed as `landcover-train`,
`landcover-evaluate` and `landcover-predict`.

## Tests

```bash
pytest
```

The suite is offline and takes a few seconds. It builds a tiny synthetic
dataset in a temporary directory, which the pipeline picks up through the
`LANDCOVER_DATA_DIR` and `LANDCOVER_RESULTS_DIR` environment variables, and
checks that the split is stratified, that the transforms produce the expected
shapes and normalisation, that the model's frozen parameter count is what the
README claims, that the metrics agree with a hand computed confusion matrix, and
that a full train, evaluate, predict round trip runs without touching the real
data or the network.

## Continuous integration

Two workflows run on every push and pull request.

`ci.yml` installs the project on a clean runner, formats and lints it with
`ruff`, runs the type check, and runs the test suite. `test.yml` runs the test
suite on all supported Python versions with the lowest and highest pinned
dependency versions, so a change that only works on one combination is caught.

Neither workflow trains a model: a training run costs hours and the test suite
covers the same code paths on synthetic data in seconds.

## Reproducing the numbers

```bash
python -m landcover.data --force
python -m landcover.train --head-epochs 5 --skip-finetune --max-train-images 4000
python -m landcover.evaluate
python -m landcover.plot
```

The run is deterministic given the seed. `config.py` records the seed, the
hyperparameters and the environment, and `results/train_history.json` records
the per epoch losses and accuracies that produced the learning curve in
[docs/figures/history.png](docs/figures/history.png).

## Known limitations

**This run was done on CPU and is smaller than intended.** The machine available
had 2 usable CPU threads and no GPU. At 628 milliseconds per image for the head
stage, the full 21,600 image training set at the originally intended 6 plus 4
epochs would have taken about 67 hours. The committed run trains on a class
stratified 4,000 image subset for 5 epochs instead, which takes about 3.2 hours
and reaches 92.2 percent. The unfreezing stage is measured at 1856 milliseconds
per image and was not included for the same reason.

The reported number is therefore a test accuracy on a subset trained model, not
the number the full configuration would reach. With the full dataset and a GPU
the same code with the same seed would be expected to land higher, and the
figures in this README were not cherry picked from a larger sweep.

**The classes overlap physically.** As described in [Results](#results), the
confusion between `HerbaceousVegetation` and `Pasture` is a property of the
labelling, not a defect of the model. Both include land covered in grass.

**The RGB dataset is a subset of EuroSAT.** EuroSAT also ships a 13 band
multispectral version. The RGB version was used here because it is small enough
to train on CPU and it is the version transfer from ImageNet works best on. The
shortwave infrared bands would be expected to improve the crop and vegetation
classes, and are a natural next step.

**Accuracy is not uniform over the world.** The dataset samples a fixed set of
locations. A model that does well on it can still do worse on imagery from a
different sensor, a different season or a different continent.
