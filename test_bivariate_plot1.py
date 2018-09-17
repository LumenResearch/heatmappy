__author__ = 'navid'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


##########################################################
# Create population form covariance and mean
mean= np.array([0, 0])
cov = np.array([[.0,-.3],[-.3,1.5]])
n_samples = 100


data = np.random.multivariate_normal(mean, cov, size=n_samples)


x = data.T[0]
y = data.T[1]
x_range = [np.floor(np.min(x))-1, np.ceil(np.max(x))+1]
y_range = [np.floor(np.min(y))-1, np.ceil(np.max(y))+1]
x_range
y_range
dbg = 1
##########################################################


##########################################################
# Create random population
# x_range = [-1, 1]
# y_range = [-5, 10]
# n_samples = 100
#
# x = np.random.rand(n_samples) * np.sum(np.abs(x_range)) + x_range[0]
# y = np.random.rand(n_samples) * np.sum(np.abs(y_range)) + y_range[0]
# data = np.array([x, y]).T
##########################################################


mean = np.mean(data.T,1)
cov = np.cov(data.T)

X = np.linspace(x_range[0], x_range[1], n_samples)
Y = np.linspace(y_range[0], y_range[1], n_samples)

pdf = np.zeros(data.shape[0])
cons = 1./((2*np.pi)**(data.shape[1]/2.)*np.linalg.det(cov)**(-0.5))
X, Y = np.meshgrid(X, Y)


cov = np.cov(data.T)

w, v = np.linalg.eig(cov)
print(w)
print(v)

def pdf(point):
  return cons*np.exp(-np.dot(np.dot((point-mean),np.linalg.inv(cov)),(point-mean).T)/2.)
zs = np.array([pdf(np.array(ponit)) for ponit in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# plot population
fig1= plt.figure(1)

plt.plot(x, y, 'ro')
plt.axis([-2,2,-10,10])
fig1.show()





# Create a surface plot and projected filled contour plot under it.
fig2= plt.figure(2)
ax = fig2.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()