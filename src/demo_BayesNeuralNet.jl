####################################
## Demo script for Bayesian neural network.

using PyPlot, PyCall

push!(LOAD_PATH, ".")
import BayesNeuralNet

"""
Sample neural nets from prior.
"""
function sample_test()
    # model parameters
    D = 1 # output
    K = 3 # hidden
    M = 2 # input
    sigma2_w = 10.0
    sigma2_y = 0.1
    
    xmin = -5
    xmax = 5
    N_lin = 1000
    X_lin = ones(M, N_lin)
    X_lin[1,:] = linspace(xmin, xmax, N_lin)
    X_lin[2,:] = 1 # bias

    # visualize
    num_samples = 5
    figure("Function samples")
    clf()
    for i in 1 : num_samples
        _, Y_true, _, _ = BayesNeuralNet.sample_data_from_prior(X_lin, sigma2_w, sigma2_y, D, K)
        plot(X_lin[1,:], Y_true)
        xlim([xmin, xmax])
    end
    ratey = (ylim()[2] - ylim()[1]) * 0.1
    ratex = (xlim()[2] - xlim()[1]) * 0.1
    text(xlim()[1] + ratex, ylim()[2] - ratey, @sprintf("K=%d", K), fontsize=18)
    show()
end

"""
Run a test script of variational inference for Bayesian neural net.
"""
function test()
    #################
    # prepara data
    
    # data size
    D = 1 # output
    M = 2 # input
        
    # function setting
    xmin = -2
    xmax = 4
    N_lin = 1000
    X_lin = ones(M, N_lin)
    X_lin[1,:] = linspace(xmin, xmax, N_lin)
    X_lin[2,:] = 1 # bias

    # training data
    N = 50 # data size
    X = 2*rand(M, N) - 0.0 # input
    X[2,:] = 1.0 # bias
    Y = 0.5*sin.(2*pi * X[1,:]/3) + 0.05 * randn(N)    
    
    # model parameters
    K = 5
    sigma2_w = 10.0
    sigma2_y = 0.01
    
    ################
    # inference
    alpha = 1.0e-5
    max_iter = 100000
    mu1, rho1, mu2, rho2 = BayesNeuralNet.VI(Y, X, sigma2_w, sigma2_y, K, alpha, max_iter)
    Y_mean = [mu2'* tanh.(mu1'X_lin[:,n]) for n in 1 : N_lin]

    ################
    # visualize        
    figure("result")
    clf()
    Y_list = []
    num_samples = 100
    for i in 1 : num_samples
        Y_est, _ = BayesNeuralNet.sample_data_from_posterior(X_lin, mu1, rho1, mu2, rho2, sigma2_y, D)
        push!(Y_list, Y_est)
        plot(X_lin[1,:], Y_est, "-c", alpha=0.25)
    end
    plot(X[1,:], Y, "ok")
    plot(X_lin[1,:], Y_mean, "b-")
    xlim([xmin, xmax])
    xlabel("x")
    ylabel("y")
    show()
end

#sample_test()
test()
