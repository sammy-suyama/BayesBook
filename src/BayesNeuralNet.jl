"""
Variational inference for Bayesian neural network
"""
module BayesNeuralNet
using Distributions

export sample_data_from_prior, sample_data_from_posterior
export VI

function sigmoid(x)
    return 1.0 / (1.0 + exp.(-x[1]))
end

function rho2sig(rho)
    return log.(1 + exp.(rho))
end

function compute_df_dmu(mu, rho, W)
    return (W - mu) ./ rho2sig(rho).^2
end

function compute_df_drho(Y, X, mu, rho, W)
    return -0.5*((W - mu).^2 - rho2sig(rho).^2) .* compute_dprec_drho(rho)
end

function compute_dprec_drho(rho)
    return 2 * rho2sig(rho) .^ (-3) .* (1 ./ (1+exp.(rho))).^2 .* (1 ./ (1+exp.(-rho)))
end

function compute_df_dw(Y, X, sigma2_y, sigma2_w, mu1, rho1, W1, mu2, rho2, W2)
    M, N = size(X)
    Y_err1 = zeros(size(W1)) # MxK
    Y_err2 = zeros(size(W2)) # KxD

    for n in 1 : N
        Z = tanh.(W1'*X[:,n]) # Kx1
        Y_est = W2'*Z
        # 2nd unit, Dx1
        delta2 = Y_est - Y[n]
        
        # 1st unit, KxD
        delta1 = diagm(1 - Z.^2) * W2 * delta2
        
        Y_err1 += X[:,n] * delta1'
        Y_err2 += Z * delta2'
    end
    df_dw1 = W1/sigma2_w + (mu1 - W1) ./ rho2sig(rho1).^2 + Y_err1 / sigma2_y
    df_dw2 = W2/sigma2_w + (mu2 - W2) ./ rho2sig(rho2).^2 + Y_err2 / sigma2_y
    return df_dw1, df_dw2
end

"""
Sample data given prior and inputs.
"""
function sample_data_from_prior(X, sigma2_w, sigma2_y, D, K)
    M, N = size(X)

    W1 = sqrt(sigma2_w) * randn(M, K)
    W2 = sqrt(sigma2_w) * randn(K, D)
    
    # sample function
    Y = [W2'* tanh.(W1'X[:,n]) for n in 1 : N]

    # sample data
    Y_obs = [W2'* tanh.(W1'X[:,n]) + sqrt(sigma2_y)*randn(D) for n in 1 : N]

    return Y_obs, Y, W1, W2
end

"""
Sample data given posterior and inputs.
"""
function sample_data_from_posterior(X, mu1, rho1, mu2, rho2, sigma2_y, D)
    N = size(X, 2)
    ep1 = randn(size(mu1))
    W1_tmp = mu1 + log.(1 + exp.(rho1)) .* ep1
    ep2 = randn(size(mu2))
    W2_tmp = mu2 + log.(1 + exp.(rho2)) .* ep2    
    Y_est = [W2_tmp'* tanh.(W1_tmp'X[:,n]) for n in 1 : N]
    Y_obs = [W2_tmp'* tanh.(W1_tmp'X[:,n]) + sqrt(sigma2_y)*randn(D)  for n in 1 : N]
    return Y_est, Y_obs
end

"""
Compute variational parameters.
"""
function VI(Y, X, sigma2_w, sigma2_y, K, alpha, max_iter)
    M, N = size(X)
    D = length(Y[1])

    # initialize
    mu1 = randn(M, K)
    rho1 = randn(M, K)
    mu2 = randn(K, D)
    rho2 = randn(K, D)

    for i in 1 : max_iter
        # sample
        ep1 = randn(size(mu1))
        W1_tmp = mu1 + log.(1 + exp.(rho1)) .* ep1
        ep2 = randn(size(mu2))
        W2_tmp = mu2 + log.(1 + exp.(rho2)) .* ep2
        
        # calc error
        df_dw1, df_dw2 = compute_df_dw(Y, X, sigma2_y, sigma2_w, mu1, rho1, W1_tmp, mu2, rho2, W2_tmp)
        
        # 1st unit
        df_dmu1 = compute_df_dmu(mu1, rho1, W1_tmp)
        df_drho1 = compute_df_drho(Y, X, mu1, rho1, W1_tmp)
        d_mu1 = df_dw1 + df_dmu1
        d_rho1 = df_dw1 .* (ep1 ./ (1+exp.(-rho1))) + df_drho1
        mu1 = mu1 - alpha * d_mu1
        rho1 = rho1 - alpha * d_rho1 
        
        # 2nd unit
        df_dmu2 = compute_df_dmu(mu2, rho2, W2_tmp)
            df_drho2 = compute_df_drho(Y, X, mu2, rho2, W2_tmp)
        d_mu2 = df_dw2 + df_dmu2
        d_rho2 = df_dw2 .* (ep2 ./ (1+exp.(-rho2))) + df_drho2
        mu2 = mu2 - alpha * d_mu2
        rho2 = rho2 - alpha * d_rho2
    end
    return mu1, rho1, mu2, rho2
end

end

