from typing import Any
import inspect
import numpy as np
from scipy.optimize import minimize, differential_evolution
import numdifftools as nd
from abc import ABC, abstractmethod


class BaseMLE(ABC):
    """
    Base class for maximum likelihood estimation.

    This class provides common functionality for fitting models with different
    noise distributions (Poisson, Gaussian, etc.). Subclasses should implement the
    negative log-likelihood and the Cramér-Rao bound or the Monte Carlo method to
    compute the covariance matrix.
    """

    def __init__(
        self,
        model,
        verbose=False,
        optimizer="nelder-mead",
        **optimizer_kwargs,
    ):
        """
        Initialize the BaseMLE.

        Parameters
        ----------
        model : callable
            The model function, f(x_data, *params).
        verbose : bool, optional
            If True, print the optimization results. Default is False.
        optimizer : str, optional
            Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
        optimizer_kwargs : dict, optional
            Additional keyword arguments passed to scipy.optimize.minimize.
        """
        self.model = self._wrap_model(model)
        self.params_init = None
        self.param_bounds = None
        self.params = None
        self.param_covs = None
        self.param_confs = None
        self.verbose = verbose
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    @staticmethod
    def _wrap_model(model):
        def model_wrapped(x_data, *params):
            y = np.asarray(model(x_data, *params))
            y = np.squeeze(y)

            return y

        return model_wrapped

    def _check_fit_args(
        self,
        x_data,
        y_data,
        params_init,
        param_bounds,
        sigma,
        is_sigma_absolute,
        quantiles=None,
    ):
        """
        Check and normalize arguments for the fit and mcfit methods.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters with size num_params. Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
            Use None for no bound. Default is None.
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.
        quantiles : tuple, optional
            The quantiles to estimate the parameter confidence intervals for.
            Default is None.

        Returns
        -------
        x_data : ndarray
            Normalized independent variable. Shape (num_vars, num_data).
        y_data : ndarray
            Dependent data. Shape (num_data,).
        params_init : array_like, optional
            Initial parameter guess. Shape (num_params,).
        param_bounds : list or None
            Normalized parameter bounds. List of tuples, each with shape (num_params,).
        sigma : ndarray
            Normalized uncertainties. Shape (num_data,).
        is_sigma_absolute : bool
            Flag for absolute sigma usage.
        quantiles : tuple or None
            Quantiles for confidence intervals.
        """
        x_data = np.atleast_2d(x_data)

        if sigma is None and is_sigma_absolute:
            raise ValueError("sigma must be provided if is_sigma_absolute=True")
        if sigma is None:
            sigma = np.ones_like(y_data)
        if np.ndim(sigma) == 0:
            sigma = np.full_like(y_data, sigma)

        if param_bounds is not None:
            param_bounds = list(zip(*param_bounds))
        if params_init is None and (
            param_bounds is None or np.any(~np.isfinite(param_bounds))
        ):
            raise ValueError(
                "Finite parameter bounds must be provided if no initial parameters are provided."
            )

        return (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma,
            is_sigma_absolute,
            quantiles,
        )

    def fit(
        self,
        x_data,
        y_data,
        params_init=None,
        param_bounds=None,
        sigma=None,
        is_sigma_absolute=False,
    ):
        """
        Perform maximum likelihood estimation fit.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data, nominally f(x_data, *params) with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters. Shape (num_params,). Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
            Use None for no bound. Default is None.
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
        """
        (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma,
            is_sigma_absolute,
            _,
        ) = self._check_fit_args(
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma,
            is_sigma_absolute,
        )

        self.params_init = params_init
        self.param_bounds = param_bounds

        self.params = self._estimate_parameters(
            x_data,
            y_data,
            sigma,
        )
        self.param_covs = self._estimate_covariances(
            x_data,
            y_data,
            sigma,
            is_sigma_absolute,
        )

        return self.params, self.param_covs

    def mcfit(
        self,
        x_data,
        y_data,
        params_init=None,
        param_bounds=None,
        sigma=None,
        is_sigma_absolute=False,
        num_samples=100,
        quantiles=(0.1, 0.9),
    ):
        """
        Perform maximum likelihood estimation fit with Monte Carlo sampling.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data, nominally f(x_data, *params) with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters with size num_params. Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
            Use None for no bound. Default is None.
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.
        num_samples : int, optional
            Number of samples to use for the Monte Carlo estimation.
            Default is 100.
        quantiles : tuple, optional
            The quantiles to estimate the parameter confidence intervals for.
            Default is (0.1, 0.9).

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
        param_confs : ndarray
            Estimated parameter confidence intervals. Shape (num_params, 2).
        """
        (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma,
            is_sigma_absolute,
            quantiles,
        ) = self._check_fit_args(
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma,
            is_sigma_absolute,
            quantiles,
        )

        self.params_init = params_init
        self.param_bounds = param_bounds

        self.params = self._estimate_parameters(
            x_data,
            y_data,
            sigma,
        )
        params_mc = self._monte_carlo_samples(
            x_data,
            y_data,
            sigma,
            is_sigma_absolute,
            num_samples,
        )
        self.param_covs = np.cov(params_mc)
        self.param_confs = np.quantile(params_mc, quantiles, axis=1)

        return self.params, self.param_covs, self.param_confs

    def predict(self, x_data):
        """
        Predict the model output for the given independent variable using the optimal parameters.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).

        Returns
        -------
        y_pred : ndarray
            Predicted dependent data. Shape (num_data,).
        """

        return self.model(x_data, *self.params)

    def _estimate_parameters(
        self,
        x_data,
        y_data,
        sigma,
    ):
        """
        Estimate the model parameters using the maximum likelihood estimation.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).
        """
        if self.params_init is None:
            result = differential_evolution(
                lambda params: self._objective(x_data, y_data, params, sigma),
                bounds=self.param_bounds,
                tol=1e-9,
                polish=False,
            )
            self.params_init = result.x

        result = minimize(
            lambda params: self._objective(x_data, y_data, params, sigma),
            x0=self.params_init,
            bounds=self.param_bounds,
            method=self.optimizer,
            **self.optimizer_kwargs,
        )

        if self.verbose:
            print("Optimal params:", result.x)
            print("Success:", result.success)
            print("Iterations:", result.nit)
            print("Function calls:", result.nfev)
            print("Message:", result.message)

        params = result.x

        return params

    def _estimate_covariances(
        self,
        x_data,
        y_data,
        sigma=None,
        is_sigma_absolute=False,
    ):
        """
        Calculate the covariance matrix using the Cramér-Rao bound.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
        """

        FIM = self._fisher_information_matrix(x_data, y_data, sigma, is_sigma_absolute)
        try:
            param_covs = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            param_covs = np.linalg.pinv(FIM)

        return param_covs

    def _fisher_information_matrix(
        self, x_data, y_data, sigma=None, is_sigma_absolute=False
    ):
        """
        Calculate the Fisher information matrix.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,).
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        FIM : ndarray
            Fisher information matrix. Shape (num_params, num_params).
        """
        scale_squared = self._scale_squared(x_data, y_data, sigma, is_sigma_absolute)

        S_sq_inv = np.diag(1 / scale_squared)
        J = self._jacobian(x_data)
        FIM = J.T @ S_sq_inv @ J

        return FIM

    def _jacobian(self, x_data):
        """
        Calculate the Jacobian matrix numerically.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).

        Returns
        -------
        J : ndarray
            Jacobian matrix. Shape (num_data, num_params).
        """
        params = self.params
        jacobian = nd.Jacobian(
            lambda p: self.model(x_data, *p), method="complex", step=1e-15
        )
        J = jacobian(params)

        return J

    def _monte_carlo_samples(
        self, x_data, y_data, sigma, is_sigma_absolute, num_samples
    ):
        """
        Calculate the Monte Carlo samples of the model parameters for variance estimation.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
        num_samples : int
            Number of samples to use for the Monte Carlo estimation.

        Returns
        -------
        params_mc : ndarray
            Monte Carlo samples of the model parameters for variance estimation.
            Shape (num_params, num_samples).
        """
        params = self.params

        num_params = len(params)

        params_mc = np.zeros((num_params, num_samples))
        for ii in range(num_samples):
            y_pred = self.predict(x_data)

            y_mc = y_pred + self._sample_noise(x_data, y_data, sigma, is_sigma_absolute)
            params_mc[:, ii] = self._estimate_parameters(x_data, y_mc, sigma)

        return params_mc

    @abstractmethod
    def _sample_noise(self, x_data, y_data, sigma, is_sigma_absolute):
        """
        Return the noise samples from the noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
        Returns
        -------
        noise : ndarray
            Noise samples from the noise distribution. Shape (num_data,).
        """
        pass

    @abstractmethod
    def _objective(self, x_data, y_data, params, sigma):
        """
        Calculate the objective function derived from the negative log-likelihood

        This method must be implemented by the subclass to define the specific
        objective function for their noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable where the data is measured.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.

        Returns
        -------
        obj : float
            Value of the objective function.
        """
        pass

    @abstractmethod
    def _scale_squared(self, x_data, y_data, sigma, is_sigma_absolute):
        """
        Calculate the squared scale parameter of the noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma : array_like, optional
            Uncertainties in y_data with shape (num_data,). May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        scale_squared : ndarray
            Squared scale parameter of the noise distribution. Shape (num_data,).
        """
        pass
