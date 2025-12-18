from dackar.anomalies import kernel_two_sample_test as K2ST

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

np.random.seed(0)

m = 200
n = 200
d = 2

sigmaX = np.eye(d)
muX = np.zeros(d)

sigmaY = np.eye(d)
muY = np.zeros(d)

iterations = 2000

X = np.random.multivariate_normal(mean=muX, cov=sigmaX, size=m)
Y = np.random.multivariate_normal(mean=muY, cov=sigmaY, size=n)

def test_kernel_two_sample_test():
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    mmd2u, mmd2u_null, p_value = K2ST.kernel_two_sample_test(X, Y,
                                                        kernel_function='rbf',
                                                        iterations=iterations,
                                                        gamma=1.0/sigma2,
                                                        verbose=True)

    assert abs(mmd2u - (-0.0032396905173185386)) < 1E-8

def test_kernel_two_sample_test():
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    mmd2u, mmd2u_null, p_value = K2ST.kernel_two_sample_test(X, Y,
                                                        kernel_function='rbf',
                                                        iterations=iterations,
                                                        gamma=1.0/sigma2,
                                                        verbose=True)

    assert abs(p_value - 0.875) < 1E-3

np.random.seed(0)

m = 200
n = 200
d = 2

sigma2X = np.eye(d)
muX = np.zeros(d)

sigma2Y = np.eye(d)
muY = np.ones(d)

iterations = 2000

X = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=m)
Y = np.random.multivariate_normal(mean=muY, cov=sigma2Y, size=n)

def test_kernel_two_sample_test():
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    mmd2u, mmd2u_null, p_value = K2ST.kernel_two_sample_test(X, Y,
                                                        kernel_function='rbf',
                                                        iterations=iterations,
                                                        gamma=1.0/sigma2,
                                                        verbose=True)

    assert abs(mmd2u - 0.19124120543500756) < 1E-8

def test_kernel_two_sample_test():
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    mmd2u, mmd2u_null, p_value = K2ST.kernel_two_sample_test(X, Y,
                                                        kernel_function='rbf',
                                                        iterations=iterations,
                                                        gamma=1.0/sigma2,
                                                        verbose=True)

    assert abs(p_value - 0.0005) < 1E-5

'''
plt.figure()
prob, bins, patches = plt.hist(mmd2u_null, bins=50) #, normed=True
plt.plot(mmd2u, prob.max()/30, 'w*', markersize=24, markeredgecolor='k',
            markeredgewidth=2, label="$MMD^2_u = %s$" % mmd2u)
plt.xlabel('$MMD^2_u$')
plt.ylabel('$p(MMD^2_u)$')
plt.legend(numpoints=1)
plt.title('$MMD^2_u$: null-distribution and observed value. $p$-value=%s'
            % p_value)
plt.show()
'''
