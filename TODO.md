# TODO

## Features

- [ ] **Add further noise models**: E.g. Rice or Rayleigh distribution.

- [ ] **Estimate absolute noise**: Estimate absolute noise by using qn_scale on the first derivate of y_data.

- [ ] **Numerical scale_squared**: Estimate the scale_squared numerically through minimization of the NLL for non-semi-analytical models.

- [ ] **Support for x-axis uncertainties (σₓ)**: Extend the MLE framework to account for measurement errors in the independent variable (x-axis), not just the dependent variable (y-axis).

- [ ] **Likelihood profiling for confidence intervals**: Implement confidence interval estimation using the likelihood profiling method as an complement to the current Fisher information matrix approach.
