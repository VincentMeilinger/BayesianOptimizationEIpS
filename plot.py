import torch
from torch import tensor
from botorch.models import SingleTaskGP
import botorch.acquisition as acqf
from matplotlib import pyplot as plt
import numpy as np
from test_functions import himmelblau

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

# BO design points
x_ = torch.arange(-1.0, 1.0, 0.01)
y_ = torch.arange(-1.0, 1.0, 0.01)
x_axis, y_axis = torch.meshgrid(x_, y_)
grid = torch.rot90(torch.stack((x_axis, y_axis), 2))

X_ = grid.reshape(len(x_)*len(y_), 2)

#%% Plotting
true_mins = tensor([
    [3., 2.],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
])

input_scaling = 5
lim_x0 = -input_scaling
lim_x1 = input_scaling
lim_y0 = -input_scaling
lim_y1 = input_scaling
def plot_bo(gp: SingleTaskGP, EI: acqf.ExpectedImprovement, X_samples):

    scores = himmelblau(x_axis, y_axis)
    p, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.title.set_text("Iteration " + str(len(X_samples)-1))
    ax.imshow(torch.rot90(scores), extent=[-5, 5, -5, 5])
    con = ax.contour(x_axis * input_scaling, y_axis * input_scaling, scores, colors="k", levels=15)
    ax.clabel(con, fontsize=9, inline=True)
    if len(X_samples) > 1:
        ax.scatter(X_samples[:, 0][0:-1]*input_scaling, X_samples[:, 1][0:-1]*input_scaling, color='white')
    ax.scatter(X_samples[:, 0][-1]*input_scaling, X_samples[:, 1][-1]*input_scaling, color='red')
    ax.scatter(true_mins[:,0], true_mins[:,1], color='red', marker='x')
    for i in range(len(X_samples)):
        ax.text(X_samples[i][0]*input_scaling + 0.1, X_samples[i][1]*input_scaling, i, color='white')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlim(lim_x0, lim_x1)
    plt.ylim(lim_y0, lim_y1)

    fig, (sub2, sub3) = plt.subplots(1, 2, figsize=(10, 5))
    #sub2 = fig.add_subplot(1, 2, 1)
    sub2.title.set_text('Acquisition Function')
    if len(X_samples) > 1:
        sub2.scatter(X_samples[:, 0][0:-1]*input_scaling, X_samples[:, 1][0:-1]*input_scaling, color='white')
    sub2.scatter(X_samples[:, 0][-1]*input_scaling, X_samples[:, 1][-1]*input_scaling, color='red')
    X_ei = X_.unsqueeze(1)
    ei = EI(X_ei).detach().numpy().reshape(len(x_), len(y_))
    sub2.imshow(ei, extent=[lim_x0, lim_x1, lim_y0, lim_y1])
    sub2.spines['left'].set_position('zero')
    sub2.spines['bottom'].set_position('zero')
    sub2.spines['right'].set_color('none')
    sub2.spines['top'].set_color('none')
    sub2.xaxis.set_ticks_position('bottom')
    sub2.yaxis.set_ticks_position('left')
    sub2.set_xlabel('gamma')
    sub2.set_ylabel('lambda')

    #sub3 = fig.add_subplot(1, 2, 2)
    sub3.title.set_text('GP posterior')
    posterior_mean = gp.posterior(X_).mean.detach().numpy().reshape(len(x_), len(y_))
    sub3.imshow(posterior_mean, extent=[lim_x0, lim_x1, lim_y0, lim_y1])
    sub3.spines['left'].set_position('zero')
    sub3.spines['bottom'].set_position('zero')
    sub3.spines['right'].set_color('none')
    sub3.spines['top'].set_color('none')
    sub3.xaxis.set_ticks_position('bottom')
    sub3.yaxis.set_ticks_position('left')
    sub3.set_xlabel('gamma')
    sub3.set_ylabel('lambda')

    plt.show()

    #fig = plt.figure(figsize=plt.figaspect(0.5))
    #ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    #ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1.plot_wireframe(x_axis*input_scaling, y_axis*input_scaling, ei, cmap ='viridis', edgecolor ='green')
    #ax2.plot_wireframe(x_axis*input_scaling, y_axis*input_scaling, posterior_mean, cmap ='viridis', edgecolor ='green')

    #plt.show()

def plot_test_func():
    scores = himmelblau(x_axis, y_axis)
    p, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(torch.rot90(scores), extent=[lim_x0, lim_x1, lim_y0, lim_y1])
    con = ax.contour(x_axis * input_scaling, y_axis * input_scaling, scores, colors="k", levels=12)
    ax.clabel(con, fontsize=9, inline=True)
    ax.scatter(true_mins[:, 0], true_mins[:, 1], color='red', marker='x')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlim(lim_x0, lim_x1)
    plt.ylim(lim_y0, lim_y1)
    plt.plot()

    p = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')

    scores = himmelblau(x_axis, y_axis)
    ax.plot_wireframe(x_axis * input_scaling, y_axis * input_scaling, scores, cmap='viridis', edgecolor='green')
    plt.show()


def plot_prediction(X_train, y_train, X, y, gamma, llambda):
    xx, yy = np.meshgrid(np.linspace(-3, 5, 200), np.linspace(-3, 5, 200))

    kernel = RBF(length_scale=gamma)
    kr = KernelRidge(alpha=llambda, gamma=gamma, kernel=kernel)
    kr.fit(X_train, y_train)

    Z = kr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    p, ax = plt.subplots(1, 1, figsize=(10, 10))
    X0 = X[y == 0]
    X1 = X[y == 1]
    ax.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu, shading='nearest')
    ax.scatter(X0[:, 0], X0[:, 1], c='blue')
    ax.scatter(X1[:, 0], X1[:, 1], c='red')

    plt.plot()

def plot_KRR_space(scores, X_samples, gamma_space, llambda_space):
    X_samples = torch.exp2(X_samples)
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        #norm=MidpointNormalize(vmin=0.1, midpoint=0.54),
    )
    plt.xlabel("gamma")
    plt.ylabel("lambda")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_space)), gamma_space, rotation=45)
    plt.yticks(np.arange(len(llambda_space)), llambda_space)
    plt.title("Validation accuracy")
    plt.scatter(X_samples[:, 0], X_samples[:, 1], c='green')
    plt.show()

def plot_bo_KRR(gp: SingleTaskGP, EI: acqf.ExpectedImprovement, X_samples):
    fig, (sub2, sub3) = plt.subplots(1, 2, figsize=(10, 5))
    #sub2 = fig.add_subplot(1, 2, 1)
    sub2.title.set_text('Acquisition Function')
    X_ei = X_.unsqueeze(1)
    ei = EI(X_ei).detach().numpy().reshape(len(x_), len(y_))
    sub2.imshow(ei, extent=[0, 1, 0, 1])
    sub2.spines['left'].set_position('zero')
    sub2.spines['bottom'].set_position('zero')
    sub2.spines['right'].set_color('none')
    sub2.spines['top'].set_color('none')
    sub2.xaxis.set_ticks_position('bottom')
    sub2.yaxis.set_ticks_position('left')
    sub2.set_xlabel('gamma')
    sub2.set_ylabel('lambda')

    #sub3 = fig.add_subplot(1, 2, 2)
    sub3.title.set_text('GP posterior')
    posterior_mean = gp.posterior(X_).mean.detach().numpy().reshape(len(x_), len(y_))
    sub3.imshow(posterior_mean, extent=[0, 1, 0, 1])
    sub3.spines['left'].set_position('zero')
    sub3.spines['bottom'].set_position('zero')
    sub3.spines['right'].set_color('none')
    sub3.spines['top'].set_color('none')
    sub3.xaxis.set_ticks_position('bottom')
    sub3.yaxis.set_ticks_position('left')
    sub3.set_xlabel('gamma')
    sub3.set_ylabel('lambda')

    plt.show()