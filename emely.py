import numpy as np
from scipy.optimize import minimize
from scipy.differentiate import jacobian


def mle_fit(
    f,
    x_data,
    y_data,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    noise_type="poisson",
    method="nelder-mead",
    bounds=None,
    **kwargs,
):
    """
    Maximum Likelihood Estimation fit function compatible with scipy.optimize.curve_fit interface.

    Parameters
    ----------
    f : callable
        The model function, f(x_data, *p). Must take the independent variable
        as the first argument and the parameters to fit as separate remaining arguments.
    x_data : array_like
        The independent variable with shape (n_v, n_x).
    y_data : array_like
        The dependent data, nominally f(x_data, *p) with size n_x.
    p0 : array_like, optional
        Initial guess for the parameters with size n_p. Default is None.
    sigma : array_like, optional
        Uncertainties in y_data. Only used for Gaussian noise type.
    absolute_sigma : bool, optional
        If True, the uncertainties are absolute. If False, the uncertainties are relative. Default is False.
    noise_type : str, optional
        Noise type: "poisson" or "gaussian". Default is "poisson".
    method : str, optional
        Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
    bounds : array_like, optional
        Bounds for the parameters as (lower_bounds, upper_bounds) with size n_p for the lower and upper bounds, respectively.
        Use None for no bound. Default is None.
    **kwargs
        Additional keyword arguments passed to scipy.optimize.minimize.

    Returns
    -------
    p_opt : array
        Optimal values for the n_p parameters so that the negative log-likelihood
        is minimized.
    p_cov : 2-D array
        The estimated covariance matrix of p_opt of shape (n_p, n_p).
    """
    x_data = np.atleast_2d(x_data)

    noise_type = noise_type.lower()

    bounds = list(zip(*bounds)) if bounds is not None else None

    if sigma is None and absolute_sigma:
        raise ValueError("sigma must be provided if absolute_sigma=True")

    result = minimize(
        lambda p: negative_log_likelihood(f, x_data, y_data, p, sigma, noise_type),
        x0=p0,
        bounds=bounds,
        method=method,
        **kwargs,
    )

    p_opt = result.x
    p_cov = covariance_matrix(
        f, x_data, y_data, p_opt, sigma, absolute_sigma, noise_type
    )

    return p_opt, p_cov


def negative_log_likelihood(f, x_data, y_data, p, sigma, noise_type):
    """
    Calculate the negative log-likelihood for MLE estimation.

    Parameters
    ----------
    f : callable
        The model function, f(x_data, *params).
    x_data : array_like
        The independent variable where the data is measured.
    y_data : array_like
        The dependent data.
    p : array_like
        Parameter values.
    sigma : array_like, optional
        Uncertainties in y_data. Only used for Gaussian noise type.
    noise_type : str
        Noise type: "poisson" or "gaussian".

    Returns
    -------
    nll : float
        Negative log-likelihood value.
    """
    y_pred = f(x_data, *p)
    y_pred = np.clip(y_pred, 1e-12, np.inf)

    if noise_type == "poisson":
        nll = -np.sum(y_data * np.log(y_pred) - y_pred)
    elif noise_type == "gaussian":
        if sigma is None:
            nll = np.sum((y_data - y_pred) ** 2)
        else:
            nll = np.sum((y_data - y_pred) ** 2 / sigma**2)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return nll


def covariance_matrix(f, x_data, y_data, p, sigma, absolute_sigma, noise_type):
    """
    Calculate the covariance matrix using the Fisher information matrix.

    Parameters
    ----------
    f : callable
        The model function.
    x : array_like
        The independent variable.
    y : array_like
        The dependent data.
    p : array_like
        Optimal parameter values.
    sigma : array_like, optional
        Uncertainties in y_data. Only used for Gaussian noise type.
    absolute_sigma : bool, optional
        If True, the uncertainties are absolute. If False, the uncertainties are relative. Default is False.
    noise_type : str
        Noise type: "poisson" or "gaussian".

    Returns
    -------
    covariance : 2-D array
        The estimated covariance matrix.
    """
    y_pred = f(x_data, *p)
    y_pred = np.clip(y_pred, 1e-12, np.inf)

    n_v, n_x = x_data.shape
    n_p = len(p)

    if noise_type == "poisson":
        w = 1 / y_pred
    elif noise_type == "gaussian":
        if absolute_sigma:
            w = 1 / sigma**2
        else:
            w = (n_x - n_p) / np.sum((y_data - y_pred) ** 2)

        w = np.full_like(y_pred, w)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    W = np.diag(w.flatten())

    J = []
    for ii in range(n_x):
        J.append(jacobian(lambda p: f(x_data[:, ii], *p), p, initial_step=1e-9).df)

    J = np.array(J)

    FIM = J.T @ W @ J
    try:
        covariance = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(FIM)

    return covariance
