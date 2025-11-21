import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import median_abs_deviation
import numdifftools as nd
from abc import ABC, abstractmethod


class BaseMLE(ABC):
    """
    Base class for maximum likelihood estimation.

    This class provides common functionality for fitting models with different
    noise distributions (Poisson, Gaussian, Laplace, etc.). Subclasses should implement the
    negative log-likelihood and the Cramér-Rao bound to compute the covariance matrix.
    """

    def __init__(
        self,
        model,
        verbose=False,
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
        **optimizer_kwargs
            Keyword arguments passed to scipy.optimize.minimize. Default values:
            - method : str, default "nelder-mead"
                Optimization method to use.
            - tol : float, default 1e-9
                Tolerance for termination.
            Any additional keyword arguments are also passed through to
            scipy.optimize.minimize. User-provided values override the defaults.
        """
        self.model = self._wrap_model(model)
        self.params_init = None
        self.param_bounds = None
        self.params = None
        self.param_covs = None
        self.verbose = verbose

        default_optimizer_kwargs = {
            "method": "nelder-mead",
            "tol": 1e-9,
        }
        self.optimizer_kwargs = {**default_optimizer_kwargs, **optimizer_kwargs}

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
        sigma_y,
        is_sigma_y_absolute,
    ):
        """
        Check and normalize arguments for the fit method.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters. Shape (num_params,). Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
            Use None for no bound. Default is None.
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

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
        sigma_y : ndarray
            Uncertainties (standard deviation) in y_data with shape (num_data,).
        is_sigma_y_absolute : bool
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
        """
        x_data = np.atleast_2d(x_data)

        if sigma_y is None and is_sigma_y_absolute:
            raise ValueError("sigma_y must be provided if is_sigma_y_absolute=True")

        if sigma_y is None:
            sigma_y = np.ones_like(y_data, dtype=float)

        if np.ndim(sigma_y) == 0:
            sigma_y = np.full_like(y_data, sigma_y, dtype=float)

        if not is_sigma_y_absolute:
            sigma_y /= np.mean(sigma_y)

            dy_data = np.diff(y_data)
            weight = 1.4826 * median_abs_deviation(dy_data, scale=1)

        num_params = None

        if params_init is not None:
            num_params = len(params_init)

        if param_bounds is not None:
            param_bounds = list(zip(*param_bounds))
            if num_params is None:
                num_params = len(param_bounds)

        if num_params is None:
            raise ValueError(
                "Either initial parameters or parameter bounds must be provided."
            )

        if params_init is None:
            if not np.all(np.isfinite(param_bounds)):
                raise ValueError(
                    "Finite parameter bounds must be provided if no initial parameters are provided."
                )
            params_init = [None] * num_params

        if param_bounds is None:
            param_bounds = [(None, None)] * num_params

        if not self.is_semi_analytical and not is_sigma_y_absolute:
            params_init = list(params_init) + [weight]
            param_bounds = list(param_bounds) + [(1e-3 * weight, 1e3 * weight)]

        return (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma_y,
            is_sigma_y_absolute,
        )

    def fit(
        self,
        x_data,
        y_data,
        params_init=None,
        param_bounds=None,
        sigma_y=None,
        is_sigma_y_absolute=False,
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
            If None, parameters are initialized using stochastic search (differential_evolution).
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds), each with shape (num_params,).
            Use None for no bound. Default is None.
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
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
            sigma_y,
            is_sigma_y_absolute,
        ) = self._check_fit_args(
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma_y,
            is_sigma_y_absolute,
        )

        self.params_init = params_init
        self.param_bounds = param_bounds

        self.params = self._estimate_parameters(
            x_data,
            y_data,
            sigma_y,
            is_sigma_y_absolute,
        )
        self.param_covs = self._estimate_covariances(
            x_data,
            y_data,
            sigma_y,
            is_sigma_y_absolute,
        )

        if not self.is_semi_analytical and not is_sigma_y_absolute:
            self.params = self.params[:-1]
            self.param_covs = self.param_covs[:-1, :-1]

        return self.params, self.param_covs

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
        y_cov : ndarray
            Covariance matrix of the predicted dependent data. Shape (num_data, num_data).
        """

        y_pred = self.model(x_data, *self.params)

        model = lambda params: self.model(x_data, *params)
        J = nd.Jacobian(model, method="complex", step=1e-15)(self.params)

        y_cov = J @ self.param_covs @ J.T

        return y_pred, y_cov

    def _estimate_parameters(
        self,
        x_data,
        y_data,
        sigma_y,
        is_sigma_y_absolute,
    ):
        """
        Estimate the model parameters using the maximum likelihood estimation.

        If params_init is None, parameters are first initialized using stochastic search
        (differential_evolution) before refinement with the standard optimizer.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).
        """
        objective = lambda params: self._negative_log_likelihood(
            x_data, y_data, params, sigma_y, is_sigma_y_absolute
        )

        if np.any([p_i is None for p_i in self.params_init]):
            if self.verbose:
                print("Estimating initial parameters...")

            result = differential_evolution(
                objective,
                bounds=self.param_bounds,
                tol=self.optimizer_kwargs["tol"],
                polish=False,
            )

            if self.verbose:
                print("Initial parameters:", result.x)
                print("Success:", result.success)
                print("Iterations:", result.nit)
                print("Function calls:", result.nfev)
                print("Message:", result.message)
                print("--------------------------------")

            self.params_init = result.x

        if self.verbose:
            print("Estimating optimal parameters...")

        result = minimize(
            objective,
            x0=self.params_init,
            bounds=self.param_bounds,
            **self.optimizer_kwargs,
        )

        if self.verbose:
            print("Optimal parameters:", result.x)
            print("Success:", result.success)
            print("Iterations:", result.nit)
            print("Function calls:", result.nfev)
            print("Message:", result.message)
            print("--------------------------------")

        params = result.x

        return params

    def _estimate_covariances(
        self,
        x_data,
        y_data,
        sigma_y=None,
        is_sigma_y_absolute=False,
    ):
        """
        Calculate the covariance matrix using the Cramér-Rao bound.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
        """

        FIM = self._fisher_information_matrix(
            x_data, y_data, sigma_y, is_sigma_y_absolute
        )
        try:
            param_covs = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            param_covs = np.linalg.pinv(FIM)

        return param_covs

    def _fisher_information_matrix(
        self, x_data, y_data, sigma_y=None, is_sigma_y_absolute=False
    ):
        """
        Calculate the Fisher information matrix.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        FIM : ndarray
            Fisher information matrix. Shape (num_params, num_params).
        """
        if self.is_semi_analytical:
            scale_squared = self._scale_squared(
                x_data, y_data, sigma_y, is_sigma_y_absolute
            )
            S_sq_inv = np.diag(1 / scale_squared)

            model = lambda params: self.model(x_data, *params)

            J = nd.Jacobian(model, method="complex", step=1e-15)(self.params)
            FIM = J.T @ S_sq_inv @ J
        else:
            nll = lambda params: self._negative_log_likelihood(
                x_data, y_data, params, sigma_y, is_sigma_y_absolute
            )

            # logpdf doesn't accept complex numbers
            H = nd.Hessian(nll, method="central", step=np.sqrt(np.finfo(float).eps))(
                self.params
            )
            FIM = H

        return FIM

    @abstractmethod
    def _negative_log_likelihood(
        self, x_data, y_data, params, sigma_y, is_sigma_y_absolute
    ):
        """
        Calculate the negative log-likelihood.

        This method must be implemented by the subclass to define the specific
        negative log-likelihood for their noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable where the data is measured.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        nll : float
            Value of the negative log-likelihood.
        """
        pass

    @abstractmethod
    def _scale_squared(self, x_data, y_data, sigma_y, is_sigma_y_absolute):
        """
        Calculate the squared scale parameter of the noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        scale_squared : ndarray
            Squared scale parameter of the noise distribution. Shape (num_data,).
        """
        pass

    @property
    @abstractmethod
    def is_semi_analytical(self):
        """
        Indicates whether the noise model supports a semi-analytical computation of the
        Fisher Information Matrix. If True, the FIM is evaluated using

            Jᵀ @ diag(1 / s^2) @ J,

        where J is the numerical Jacobian of the model. If False, the FIM is obtained
        via a numerical Hessian of the negative log-likelihood requiring is_sigma_y_absolute=True.

        Returns
        -------
        bool
            True if semi-analytical FIM computation is supported, False otherwise.
        """
        pass
