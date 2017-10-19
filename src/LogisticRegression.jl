"""
Variational inference for Bayesian logistic regression.
"""
module LogisticRegression
using Distributions

export sigmoid, sample_data, VI

function sigmoid(x)
    return 1.0 / (1.0 + exp.(-x[1]))
end    

function bern_sample(mu)
    i = rand(Bernoulli(mu))
    val = zeros(2)
    val[i+1] = 1
    return val
end

"""
Sample data & parameter given covariance Sigma_w and inputs X.
"""
function sample_data(X, Sigma_w)
    N = size(X, 2)
    M = size(Sigma_w, 1)

    # sample parameters
    W = rand(MvNormal(zeros(M), Sigma_w))
    
    # sample data
    Y = [rand(Bernoulli(sigmoid(W'*X[:, n]))) for n in 1 : N]
    return Y, W
end

"""
Compute variational parameters.
"""
function VI(Y, X, M, Sigma_w, alpha, max_iter)
    function rho2sig(rho)
        return log.(1 + exp.(rho))
    end
    
    function compute_df_dw(Y, X, Sigma_w, mu, rho, W)
        M, N = size(X)
        term1 = (mu - W) ./ rho2sig(rho).^2
        term2 = inv(Sigma_w)*W
        term3 = 0
        for n in 1 : N
            term3 += -(Y[n] - sigmoid(W'*X[:,n])) * X[:,n]
        end
        return term1 + term2 + term3
    end
    
    function compute_df_dmu(mu, rho, W)
        return (W - mu) ./ rho2sig(rho).^2
    end
    
    function compute_df_drho(Y, X, Sigma_w, mu, rho, W)
        return -0.5*((W - mu).^2 - rho2sig(rho).^2) .* compute_dprec_drho(rho)
    end
    
    function compute_dprec_drho(rho)
        return 2 * rho2sig(rho) .^ (-3) .* (1 ./ (1+exp.(rho))).^2 .* (1 ./ (1+exp.(-rho)))
    end

    # diag gaussian for approximate posterior
    mu = randn(M)
    rho = randn(M) # sigma = log.(1 + exp.(rho))
    
    for i in 1 : max_iter
        # sample epsilon
        ep = rand(M)
        W_tmp = mu + log.(1 + exp.(rho)) .* ep

        # calculate gradient
        df_dw = compute_df_dw(Y, X, Sigma_w, mu, rho, W_tmp)
        df_dmu = compute_df_dmu(mu, rho, W_tmp)
        df_drho = compute_df_drho(Y, X, Sigma_w, mu, rho, W_tmp)
        d_mu = df_dw + df_dmu
        d_rho = df_dw .* (ep ./ (1+exp.(-rho))) + df_drho

        # update variational parameters
        mu = mu - alpha * d_mu
        rho = rho - alpha * d_rho 
    end
    return mu, rho
end

end
