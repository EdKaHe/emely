import numpy as np
from .base import BaseMLE
from scipy.stats import poisson
from scipy.special import gammaln


class PoissonMLE(BaseMLE):
    """
    Maximum likelihood estimation for Poisson noise distribution.

    This class implements MLE fitting assuming the data follows a Poisson
    distribution.

    Attributes
    ----------
    _is_semi_analytical : bool
        Indicates whether the noise model supports a semi-analytical computation of the
        Fisher Information Matrix. If True, the FIM is evaluated using

            Jᵀ @ diag(1 / s^2) @ J,

        where J is the numerical Jacobian of the model. If False, the FIM is obtained
        via a numerical Hessian of the negative log-likelihood requiring is_sigma_y_absolute=True.
    """

    _is_semi_analytical = True

    def _negative_log_likelihood(self, x_data, y_data, params, sigma_y):
        """
        Calculate the negative log-likelihood for Poisson noise.

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

        Returns
        -------
        nll : float
            Value of the negative log-likelihood.
        """
        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        # gammaln(y + 1) = log(factorial(y))
        nll = np.sum(y_pred + gammaln(y_data + 1) - y_data * np.log(y_pred))

        return nll

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
            If True, sigma_y is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        scale_squared : ndarray
            Squared scale parameter of the noise distribution. Shape (num_data,).
        """
        params = self.params

        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        return y_pred
