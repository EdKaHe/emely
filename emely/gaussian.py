import numpy as np
from .base import BaseMLE
from scipy.stats import norm


class GaussianMLE(BaseMLE):
    """
    Maximum likelihood estimation for Gaussian noise distribution.

    This class implements MLE fitting assuming the data follows a Gaussian
    (normal) distribution.
    """

    @property
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
            True, indicating semi-analytical FIM computation is supported.
        """
        return True

    def _negative_log_likelihood(
        self, x_data, y_data, params, sigma_y, is_sigma_y_absolute
    ):
        """
        Calculate the negative log-likelihood for Gaussian noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        params : array_like
            Parameter values. Shape (num_params,).
        sigma_y : array_like
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.

        Returns
        -------
        nll : float
            Value of the negative log-likelihood.
        """
        y_pred = self.model(x_data, *params)

        nll = 0.5 * np.sum(
            (y_data - y_pred) ** 2 / sigma_y**2 + np.log(2 * np.pi * sigma_y**2)
        )

        return nll

    def _estimate_absolute_sigma_y(self, x_data, y_data):
        """
        Estimate the absolute standard deviation of the noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        ndarray
            The estimated absolute standard deviation of the noise. Shape (num_data,).
        """

        sigma_y = self._sigma_y
        is_sigma_y_absolute = self._is_sigma_y_absolute
        params = self.params

        if is_sigma_y_absolute:
            return sigma_y
        else:
            _, num_data = np.shape(x_data)
            num_params = len(params)

            y_pred = self.model(x_data, *params)

            weight_squared = (
                1
                / (num_data - num_params)
                * np.sum((y_data - y_pred) ** 2 / sigma_y**2)
            )

            sigma_y = np.sqrt(weight_squared) * sigma_y
            return sigma_y

    @property
    def _scale_squared(self):
        """
        Squared scale parameter of the noise distribution.

        Returns
        -------
        ndarray
            Squared scale parameter. Shape (num_data,).
        """
        sigma_y = self._sigma_y

        scale_squared = sigma_y**2

        return scale_squared
