from __future__ import division
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial.distance import sqeuclidean


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test


def lcg(modulus, a, c, seed):
    while True:
        seed = (a * seed + c) % modulus
        yield seed
# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    # Task 2:
    sample = np.zeros((mean.shape[0],))
    # TODO: Implement a draw from a multivariate Gaussian here
    '''
    generator = lcg(2**32,22695477,1,567)
    u1 = [float(generator.next())/2.**32 for x in range(mean.shape[0])]
    u2 = [float(generator.next())/2.**32 for x in range(mean.shape[0])]
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.*3.14 * np.array(u2))
    '''


    z = np.random.standard_normal((mean.shape[0],))
    A = np.linalg.cholesky(cov)
    sample = np.dot(A,z) + mean
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n,n))

        # Task 1:
        # TODO: Implement the covariance matrix here

        for row in range(n):
            for col in range(n):
                covMat[row,col] = self.sigma2_f**2* np.exp(-1./(2*self.length_scale**2)*np.linalg.norm(X[col,:]-X[row,:])**2)
        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        total_cov = self.k.covMatrix(self.X,Xa)

        k_x_x = total_cov[:self.n,:self.n]
        k_x_xa = total_cov[:self.n,self.n:]
        k_xa_x = total_cov[self.n:,:self.n]
        k_xa_xa = total_cov[self.n:,self.n:]
        k_xa_xa -= self.k.sigma2_n * np.identity(k_xa_xa.shape[0])

        mean_fa = np.dot(np.dot(k_xa_x,np.linalg.inv(self.K)),self.y)
        cov_fa = k_xa_xa - np.dot(np.dot(k_xa_x, np.linalg.inv(self.K)), k_x_xa)

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        mll_part1 = 0.5*np.dot(np.dot(np.transpose(self.y),np.linalg.inv(self.K)),self.y)
        #mll_part2 = 0.5*np.log(np.linalg.det(self.K)) + self.n/2*np.log(2*math.pi)
        #0.5 * np.linalg.slogdet(self.K)
        sign, logdet = np.linalg.slogdet(self.K)
        mll_part2 = 0.5 * sign * logdet + self.n / 2 * np.log(2 * math.pi)
        # Return mll
        mll = mll_part1+mll_part2
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters


        distance = np.zeros((self.n, self.n), dtype=float)
        for row in range(self.n):
            for col in range(self.n):
                distance[row, col] = np.linalg.norm(self.X[col, :] - self.X[row, :]) ** 2

        dk_sigma_f = 2* ( self.K - (self.k.sigma2_n * np.identity(self.n)) )
        dk_sigma_n = 2*self.k.sigma2_n * np.identity(self.n)
        dk_l = np.multiply((distance/self.k.length_scale**2),dk_sigma_f/2)

        alpha = np.dot(np.linalg.inv(self.K),self.y)
        part_sum = (np.dot(alpha,np.transpose(alpha))) - np.linalg.inv(self.K)

        grad_ln_sigma_f = -0.5 * np.trace(np.dot(part_sum, dk_sigma_f))
        grad_ln_sigma_n = -0.5 * np.trace(np.dot(part_sum, dk_sigma_n))
        grad_ln_length_scale = -0.5 * np.trace(np.dot(part_sum, dk_l))

        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        mse = ((ya - fbar) ** 2).mean(axis = None)
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        sig_2_xt = np.diag(cov + self.k.sigma2_n * np.identity(cov.shape[0]))

        sum = 0
        for i in range(cov.shape[0]):
            sum += (0.5 * np.log(2*math.pi*sig_2_xt[i]) + ((ya[i] - fbar[i]) ** 2)/(2*sig_2_xt[i]))

        msll = sum/cov.shape[0]

        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)
    sample = multivariateGaussianDraw(np.zeros((3000,)), np.identity(3000))
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.hist(sample,bins = 100)
    plt.show()
    params = [0.5*np.log(1),np.log(0.1),0.5*np.log(0.5)]
    rbf  = RadialBasisFunction(params)
    val = GaussianProcessRegression(np.random.standard_normal((300,300)),np.random.standard_normal((300,300)), rbf)
    test = val.optimize(params)
    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
