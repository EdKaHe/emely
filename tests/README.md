# Test Suite

The `tests/` folder contains validation notebooks organized by noise type and test scenario:

## Test 1: Basic parameter and covariance validation
- **Purpose**: Validates parameter estimates and covariance matrices with known initial parameters
- **Model**: Gaussian function

## Test 2: Fitting without initial parameters
- **Purpose**: Validates parameter estimates and covariance matrices when initial parameters are not provided
- **Model**: Gaussian function

## Test 3: Unknown absolute variance
- **Purpose**: Validates parameter estimates and covariance matrices when the absolute variance is unknown
- **Model**: Gaussian function

## Test 4: Predicted value and covariance validation
- **Purpose**: Validates the predicted value and covariance of the dependent variable
- **Model**: Lorentzian function

