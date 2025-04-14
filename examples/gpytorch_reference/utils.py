import numpy as np
import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    This class defines the exact Gaussian Process model for regression.

    Args:
    - train_x (torch.Tensor): The training input data.
    - train_y (torch.Tensor): The training target data.
    - likelihood: The likelihood function for the model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
        - x (torch.Tensor): Input data.

        Returns:
        - gpytorch.distributions.MultivariateNormal: Multivariate normal distribution over the output.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def generate_regressor(x_original, n_regressors):
    """
    Generate regressor matrix by padding the original input array with zeros
    from the left for (n_regressors - 1) positions, and rolling over the input
    array where the window size equals n_regressors.

    Args:
        x_original (array-like): The original input array.
        n_regressors (int): The number of regressors.

    Returns:
        array-like: The regressor matrix.
    """
    X = []
    x_padded = np.pad(
        x_original,
        pad_width=(n_regressors - 1, 0),
        mode="constant",
        constant_values=(0),
    )

    for _ in range(len(x_original)):
        X.append(x_padded[:n_regressors])
        x_padded = np.roll(x_padded, -1)

    return np.array(X, dtype='f')


def load_data(
    train_in_path,
    train_out_path,
    test_in_path,
    test_out_path,
    size_train: int,
    size_test: int,
    n_regressors: int,
):
    """
    Load data from specified paths, preprocess, and generate regressor matrices.

    Args:
        train_in_path (str): Path to the training input data file.
        train_out_path (str): Path to the training output data file.
        test_in_path (str): Path to the testing input data file.
        test_out_path (str): Path to the testing output data file.
        size_train (int): Size of the training dataset.
        size_test (int): Size of the testing dataset.
        n_regressors (int): Number of regressors.

    Returns:
        - X_train (torch.Tensor): Regressor matrix for training data.
        - Y_train (torch.Tensor): Target values for training data.
        - X_test (torch.Tensor): Regressor matrix for testing data.
        - Y_test (torch.Tensor): Target values for testing data.
    """
    x_train_in = np.loadtxt(train_in_path, dtype='f')[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype='f')[:size_test]

    X_train = torch.from_numpy(generate_regressor(x_train_in, n_regressors))
    X_test = torch.from_numpy(generate_regressor(x_test_in, n_regressors))

    Y_train = torch.from_numpy(np.loadtxt(train_out_path, dtype='f')[:size_train])
    Y_test = torch.from_numpy(np.loadtxt(test_out_path, dtype='f')[:size_test])

    return X_train, Y_train, X_test, Y_test


def train(model, likelihood, X_train, Y_train, training_iter=10):
    """
    Train the Gaussian process regression model.

    Args:
        model (gpytorch.models.ExactGP): The Gaussian process regression model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): The likelihood function.
        X_train (torch.Tensor): The training input data.
        Y_train (torch.Tensor): The training target data.
        training_iter (int, optional): Number of training iterations.

    Returns:
        None
    """
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -(mll(output, Y_train))
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return None


def predict_with_var(model, likelihood, X_test):
    """
    Predict the mean and variance of latent function values and observed target values.

    Args:
        model (gpytorch.models.ExactGP): The trained Gaussian process regression model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): The likelihood function.
        X_test (torch.Tensor): The test input data.

    Returns:
        - f_mean (torch.Tensor): Mean of latent function values.
        - f_var (torch.Tensor): Variance of latent function values.
    """
    model.eval()
    likelihood.eval()
    with (torch.no_grad(), \
        gpytorch.settings.fast_pred_var(False), \
        # Compute the exact posterior covariance
        gpytorch.settings.lazily_evaluate_kernels(False), \
        # Kernel matrices are computed immediately, not lazily
        gpytorch.settings.fast_computations(covar_root_decomposition=False, \
                                            # Compute exact Cholesky
                                            log_prob=False, \
                                            # Compute the log-determinant exactly (via Cholesky)
                                            solves=False)):
                                            # Use direct solver
        f_pred = model(X_test)
        f_mean = f_pred.mean
        f_var = f_pred.variance

    '''
    for future plot generation:
    # Get upper and lower confidence bounds
    observed_pred = likelihood(model(test_x))
    lower, upper = observed_pred.confidence_region()
    '''

    return f_mean, f_var

def predict(model, likelihood, X_test):
    """
    Predict the mean and variance of latent function values and observed target values.

    Args:
        model (gpytorch.models.ExactGP): The trained Gaussian process regression model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): The likelihood function.
        X_test (torch.Tensor): The test input data.

    Returns:
        - f_mean (torch.Tensor): Mean of latent function values.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred = model(X_test)
        f_mean = f_pred.mean

    '''
    for future plot generation:
    # Get upper and lower confidence bounds
    observed_pred = likelihood(model(test_x))
    lower, upper = observed_pred.confidence_region()
    '''

    return f_mean


def calculate_error(Y_test, Y_pred):
    """
    Calculate the error between the true target values and the predicted target values.

    Args:
        Y_test (torch.Tensor): True target values.
        Y_pred (torch.Tensor): Predicted target values.

    Returns:
        torch.Tensor: The error between true and predicted target values.
    """
    return torch.norm(Y_test - Y_pred)
