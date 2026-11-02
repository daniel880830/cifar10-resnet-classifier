# Data

This project uses **CIFAR-10** (60,000 32×32 colour images in 10 classes; 50,000 train /
10,000 test). The images are not stored in this repository. Fetch them with:

```bash
cifar download-data
```

The command downloads the official **binary** release
(<https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>), checks its published MD5
(`c32a1d4ab5d03f1284b67883e8d87530`), decodes the fixed-size binary records (no pickle is
involved) and writes four arrays to `data/raw/`:

| File | Shape | dtype |
|---|---|---|
| `X_train.npy` | (50000, 32, 32, 3) | uint8 |
| `y_train.npy` | (50000,) | int64 |
| `X_test.npy` | (10000, 32, 32, 3) | uint8 |
| `y_test.npy` | (10000,) | int64 |

`manifest.json` holds the SHA-256 of each array; `cifar verify-data` checks them.

Reference:

> Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
> Technical report, University of Toronto. <https://www.cs.toronto.edu/~kriz/cifar.html>
