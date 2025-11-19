# TODO

## Features

- [ ] **Add further noise models**: E.g. Rice, Rayleigh, or folded Gaussian distribution.

- [ ] **Get NLL from sum(logpdf)**: Wherever no analytical solution exists, calculate the NLL from the sum of the NLL.

- [ ] **FIM from NLL**: Calculate the FIM directly from the NLL.

- [ ] **Numerical scale_squared**: Estimate the scale_squared numerically through minimization of the NLL wherever no analytical solution exists.

- [ ] **Validate Monte Carlo methods**: Validate the Monte Carlo methods used to estimate confidence intervals.

- [ ] **Support for x-axis uncertainties (σₓ)**: Extend the MLE framework to account for measurement errors in the independent variable (x-axis), not just the dependent variable (y-axis).

- [ ] **Likelihood profiling for confidence intervals**: Implement confidence interval estimation using the likelihood profiling method as an complement to the current Fisher information matrix approach.