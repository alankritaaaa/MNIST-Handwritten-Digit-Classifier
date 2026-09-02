# MNIST Handwritten Digit Classifier (CNN)

A convolutional neural network trained on the real, full MNIST dataset-
60,000 training images and 10,000 test images of handwritten digits (0-9),
built to demonstrate CNN fundamentals: convolution, pooling, batch
normalization, dropout, and the Adam optimizer.

## Pipeline
```
Raw 28x28 pixel grids → normalize → Conv2D+BatchNorm+MaxPool (x2) → Flatten → Dropout+Dense → digit prediction (0-9)
```

## Results
- **99.08% accuracy** on the full, official 10,000-image MNIST test set
- Only 92 misclassifications out of 10,000 — and the errors are genuinely
  ambiguous cases even to a human eye (e.g. a messily-written 9 misread as a 5)

## Files
- `mnist_cnn_classifier.py` — the full training/evaluation script
- `MNIST_CNN_Classifier.ipynb` — the complete, executed walkthrough notebook with explanations and visualizations
- `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` — the official MNIST dataset files (idx format)

## How to Run
```bash
pip install -r requirements.txt
python mnist_cnn_classifier.py
```
Or open `MNIST_CNN_Classifier.ipynb` in Jupyter for the full walkthrough with visualizations.

## Architecture
| Layer | Purpose |
|---|---|
| Conv2D(32) → BatchNorm → MaxPool | Learns simple strokes/edges |
| Conv2D(64) → BatchNorm → MaxPool | Combines edges into digit-shape features |
| Flatten → Dropout(0.4) | Reduces overfitting risk at the highest-parameter point |
| Dense(128) → Dropout(0.3) | Learns the final decision boundary |
| Dense(10, softmax) | Outputs a probability per digit (0-9) |

## Design Decisions
- **Two stacked convolutional blocks** rather than one — lets the network build
  a genuine feature hierarchy (edges → shapes) instead of a single, shallow pass.
- **BatchNormalization after every Conv2D** — stabilizes training and allows
  effective use of a larger learning rate via Adam.
- **Two different Dropout rates at different depths** (0.4 after Flatten, 0.3
  before the final layer) — heavier regularization where the parameter count
  (and overfitting risk) is highest.
- **Reproducibility**: fixed random seeds across `random`, `numpy`, and `tensorflow`.

## Dataset Source
Official MNIST database (LeCun et al.), sourced via a GitHub mirror of the
original idx-format files.

## Possible Extensions
- Data augmentation (random rotation/shift) to push accuracy further
- Compare against a plain Dense-only network to quantify the CNN's advantage
- Visualize learned filters from the first Conv2D layer
- Build a simple drawing-canvas web demo for live digit recognition
