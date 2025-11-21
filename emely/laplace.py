import numpy as np
from .base import BaseMLE


class LaplaceMLE(BaseMLE):
    """
    Maximum likelihood estimation for Laplace noise distribution.

    This class implements MLE fitting assuming the data follows a Laplace
    distribution.
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
        Calculate the negative log-likelihood for Laplace noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable.
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
        y_pred = self.model(x_data, *params)

        nll = np.sum(
            np.sqrt(2) * np.abs(y_data - y_pred) / sigma_y
            + np.log(np.sqrt(2) * sigma_y)
        )

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
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
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

        scale_squared = sigma_y**2 / 2
        if not is_sigma_y_absolute:
            weight_squared = (
                2
                * (
                    1
                    / (num_data - num_params)
                    * np.sum(np.abs(y_data - y_pred) / sigma_y)
                )
                ** 2
            )

            scale_squared = scale_squared * weight_squared

            self.sigma_y = np.sqrt(weight_squared) * sigma_y
        else:
            self.sigma_y = sigma_y

        return scale_squared
