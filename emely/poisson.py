import numpy as np
from .base import BaseMLE
from scipy.stats import poisson


class PoissonMLE(BaseMLE):
    """
    Maximum likelihood estimation for Poisson noise distribution.

    This class implements MLE fitting assuming the data follows a Poisson
    distribution.
    """

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

        scale_squared = self._scale_squared(x_data, y_data, sigma, is_sigma_absolute)
        mu = scale_squared

        return poisson.rvs(mu)

    def _objective(self, x_data, y_data, params, sigma):
        """
        Calculate the objective function derived from the negative log-likelihood for Poisson noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable where the data is measured.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma : array_like, optional
            Uncertainties in y_data. May be used depending on the noise distribution.

        Returns
        -------
        obj : float
            Value of the objective function.
        """
        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        obj = -np.sum(y_data * np.log(y_pred) - y_pred)

        return obj

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
        params = self.params

        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        return y_pred
