from .gaussian import GaussianMLE
from .poisson import PoissonMLE
from .laplace import LaplaceMLE
from .folded_gaussian import FoldedGaussianMLE


def curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    bounds=None,
    sigma=None,
    absolute_sigma=False,
    method="nelder-mead",
    noise="gaussian",
    **optimizer_kwargs,
):
    """
    This function provides a scipy.optimize.curve_fit-like interface for maximum likelihood
    estimation (MLE) fitting with support for different noise distributions. It wraps the
    BaseMLE classes to provide a familiar interface to the scipy.optimize.curve_fit function.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent variable as the first
        argument and the parameters to fit as separate remaining arguments.
    xdata : array_like
        The independent variable with shape (num_vars, num_data).
    ydata : array_like
        The dependent data. Shape (num_data,).
    p0 : array_like, optional
        Initial guess for the parameters. Shape (num_params,). Default is None.
    bounds : 2-tuple of array_like, optional
        Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
        Use None for no bound. Default is None.
    sigma : array_like, optional
        Uncertainties (standard deviation) in ydata with shape (num_data,).
        May be used depending on the noise distribution.
    absolute_sigma : bool, optional
        If True, sigma is the absolute standard deviation of the noise.
        If False, the absolute standard deviation is estimated from the data.
        Default is False.
    method : str, optional
        Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
        This parameter takes precedence over any `method` specified in `optimizer_kwargs`.
    noise : {"gaussian", "poisson", "laplace", "folded-gaussian"}, optional
        Noise type for maximum likelihood estimation. Default is "gaussian".
        - "gaussian": Assumes a Gaussian (normal) noise distribution
        - "poisson": Assumes a Poisson noise distribution
        - "laplace": Assumes a Laplace noise distribution
        - "folded-gaussian": Assumes a folded Gaussian (folded normal) noise distribution
    **optimizer_kwargs
        Additional keyword arguments passed to scipy.optimize.minimize.
        Note: The `method` parameter will override any `method` key in `optimizer_kwargs`.

    Returns
    -------
    popt : ndarray
        Optimal parameter values. Shape (num_params,).
    pcov : ndarray
        Estimated covariance matrix. Shape (num_params, num_params).
    """

    model = f
    x_data = xdata
    y_data = ydata
    params_init = p0
    param_bounds = bounds
    sigma_y = sigma
    is_sigma_y_absolute = absolute_sigma
    optimizer_kwargs["method"] = method

    if noise == "gaussian":
        MLE = GaussianMLE
    elif noise == "poisson":
        MLE = PoissonMLE
    elif noise == "laplace":
        MLE = LaplaceMLE
    elif noise == "folded-gaussian":
        MLE = FoldedGaussianMLE
    else:
        raise ValueError(f'Invalid noise type "{noise}".')

    estimator = MLE(model, **optimizer_kwargs)

    params, params_cov = estimator.fit(
        x_data,
        y_data,
        params_init,
        param_bounds,
        sigma_y,
        is_sigma_y_absolute,
    )

    popt = params
    pcov = params_cov

    return popt, pcov
