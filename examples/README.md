# Examples

The `examples/` folder contains demonstration notebooks showcasing the use of the Emely package for maximum likelihood estimation with different models and noise types:

## Example 1: Basic Gaussian fitting with Poisson noise
- **Purpose**: Demonstrates parameter estimation for a normalized Gaussian function with Poisson noise
- **Model**: 1D Gaussian function
- **Noise**: Poisson statistics

## Example 2: Decaying Gaussian fitting with Poisson noise
- **Purpose**: Demonstrates parameter estimation for a normalized and decaying Gaussian function with Poisson noise
- **Model**: 2D decaying Gaussian function (spatial and temporal dimensions)
- **Noise**: Poisson statistics

## Example 3: Decaying Gaussian with diffusion
- **Purpose**: Demonstrates parameter estimation for a decaying Gaussian function with time-dependent width (diffusion) and Poisson noise
- **Model**: 2D decaying Gaussian with diffusion (spatial and temporal dimensions)
- **Noise**: Poisson statistics

## Example 4: Error validation for Gaussian fitting
- **Purpose**: Validates estimated errors against empirical errors for different Poisson noise samples using a Gaussian model
- **Model**: 1D Gaussian function
- **Noise**: Poisson statistics
- **Comparison**: Least-Squares, Gaussian-MLE, and Poisson-MLE estimators

## Example 5: Error validation for decaying Gaussian with diffusion
- **Purpose**: Validates estimated errors against empirical errors for different Poisson noise samples using a decaying Gaussian with diffusion model
- **Model**: 2D decaying Gaussian with diffusion (spatial and temporal dimensions)
- **Noise**: Poisson statistics
- **Comparison**: Least-Squares, Gaussian-MLE, and Poisson-MLE estimators

