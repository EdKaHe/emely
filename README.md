# Emely

**Emely** is a lightweight Python package for **maximum likelihood estimation (MLE)**–based parameter fitting.  
It provides a `curve_fit`-like interface built on top of `scipy.optimize.minimize`, with support for **Poisson** and **Gaussian** noise models.

---

## Features

- **MLE fitting** for Poisson and Gaussian noise  
- Mirrors the function inputs and outputs of `scipy.optimize.curve_fit`  
- Automatic **covariance matrix estimation** via the Fisher information matrix  

---

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# fit Gaussian using MLE for Poisson noise
import numpy as np
import matplotlib.pyplot as plt

from emely import mle_fit


# define the model
def gaussian(x, a, mu, sigma):
    return a / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# create the Gaussian data with Poisson noise
p = (100, 5, 1)
x_data = np.linspace(-10, 10, 1001)
y_data = np.random.poisson(gaussian(x_data, *p))

# fit using MLE for Poisson noise
p_opt, p_cov = mle_fit(
    gaussian,
    x_data,
    y_data,
    p0=(50, 10, 5),
    noise_type="poisson",
)

# show the fit
plt.plot(x_data, y_data, label="Measurement")
plt.plot(x_data, gaussian(x_data, *p_opt), label="Fit")

plt.grid()
plt.legend()
```

## Why MLE for Poisson Data?

Maximum likelihood estimation (MLE) can correctly model Poisson- or Gaussian-distributed noise, leading to more accurate and unbiased parameter estimate as compared to least-squares fitting, which is only optimal for Gaussian-distributed noise.

## Examples

See the provided notebooks for detailed usage and comparisons:
- example_1.ipynb: 1D Gaussian fitting with Poisson and Gaussian noise
- example_2.ipynb: 2D Gaussian fitting and performance benchmarking
These examples also compare least-squares, Gaussian MLE, and Poisson MLE approaches.

## License

MIT License © 2025

