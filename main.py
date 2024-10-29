import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

# Function to generate synthetic data from a Gaussian Mixture Model (GMM)
def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x

# Function to generate data samples from a GMM
def generateDataFromGMM(N, gmmParameters):
    # Generates N vector samples from the specified mixture of Gaussians
    # Returns samples and their component labels
    # Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    return x, labels

# Function to plot 3D data
def plot3(a, b, c, mark="o", col="b"):
    from matplotlib import pyplot
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    pylab.ion()
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title('Training Dataset')

# Function to transform input features for linear regression
def z(x):
    return np.asarray([np.ones(x.shape[1]), x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2, x[0] ** 3, (x[0] ** 2) * x[1],
                       x[0] * (x[1] ** 2), x[1] ** 3])

# Function to define the MLE objective function
def func_MLE(x, y):
    zx = z(x)
    
    def f(theta):
        w = theta[0:10]
        sigma = theta[10]
        return np.log(sigma) + np.mean((y - w @ zx) ** 2) / 2 / sigma ** 2
    
    return f

# Function to define the MAP objective function
def func_MAP(x, y, gamma):
    zx = z(x)
    mu = np.zeros(zx.shape[0])
    sigma_w = np.eye(mu.size) * gamma
    
    def f(theta):
        w = theta[0:10]
        sigma_v = theta[10]
        return np.log(sigma_v) + np.mean((y - w @ zx) ** 2) / 2 / sigma_v ** 2 + np.log(multivariate_normal.pdf(w, mu, sigma_w))
    
    return f

# Function to calculate the mean squared error
def loss(w, x, y):
    return np.mean((y - w @ z(x)) ** 2)

# Main function
def main():
    # Generate synthetic data for training and validation
    x_train, y_train, x_valid, y_valid = hw2q2()

    # MLE optimization
    res = minimize(func_MLE(x_train, y_train), np.random.random(11),
                   method='Nelder-Mead',
                   options={'maxiter': 10000})
    
    # Calculate MLE training and validation losses
    loss_MLE_train = np.log(loss(res.x[:-1], x_train, y_train))
    loss_MLE_valid = np.log(loss(res.x[:-1], x_valid, y_valid))
    
    # MAP optimization for different values of gamma
    loss_MAP_train = []
    loss_MAP_valid = []
    m, n = -10, 10
    
    for i in range(m, n + 1):
        print('gamma = 10^', i)
        gamma = 10 ** i
        res = minimize(func_MAP(x_train, y_train, gamma),
                       np.random.random(11),
                       method='Nelder-Mead',
                       options={'maxiter': 2000})
        loss_MAP_train.append(np.log(loss(res.x[:-1], x_train, y_train)))
        loss_MAP_valid.append(np.log(loss(res.x[:-1], x_valid, y_valid)))
    
    # Plot the results
    plt.figure()
    plt.plot(range(m, n + 1), loss_MAP_train, label='P_TRAIN')
    plt.plot(range(m, n + 1), loss_MAP_valid, label='P_VALIDATION')
    plt.plot([-15, 15], [loss_MLE_train, loss_MLE_train], label='E_TRAIN')
    plt.plot([-15, 15], [loss_MLE_valid, loss_MLE_valid], label='E_VALIDATION')
    plt.xticks(range(m, n + 1))
    plt.ylabel('$ln(loss)$')
    plt.xlabel('$log {10}\\gamma$')
    plt.legend()
    plt.savefig('MAPandMLE.png')
    plt.show()

if __name__ == "__main__":
    main()
