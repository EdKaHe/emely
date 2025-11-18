import numpy as np
from .base import BaseMLE
from scipy.stats import norm


class GaussianMLE(BaseMLE):
    """
    Maximum likelihood estimation for Gaussian noise distribution.

    This class implements MLE fitting assuming the data follows a Gaussian
    (normal) distribution.
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
        scale = np.sqrt(scale_squared)

        return norm.rvs(scale=scale)

    def _negative_log_likelihood(self, x_data, y_data, params, sigma):
        """
        Calculate the negative log-likelihood for Gaussian noise.

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
        nll : float
            Value of the negative log-likelihood.
        """
        y_pred = self.model(x_data, *params)

        nll = np.sum((y_data - y_pred) ** 2 / sigma**2)

        return nll

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

        _, num_data = np.shape(x_data)
        num_params = len(params)

        y_pred = self.model(x_data, *params)

        scale_squared = sigma**2
        if not is_sigma_absolute:
            weight = (
                1 / (num_data - num_params) * np.sum((y_data - y_pred) ** 2 / sigma**2)
            )
            scale_squared = scale_squared * weight

        return scale_squared
