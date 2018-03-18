"""
Variational inference for Bayesian DimensionalityReduction
"""
module DimensionalityReduction

using Distributions
#using ProgressMeter

export DRModel
export sample_data, VI

####################
## Types
struct DRModel
    D::Int
    M::Int
    sigma2_y::Float64
    m_W::Array{Float64, 2} # MxD
    Sigma_W::Array{Float64, 3} # MxMxD
    m_mu::Array{Float64, 1} # D
    Sigma_mu::Array{Float64, 2} # DxD
end

####################
## functions
function sqsum(mat::Array{Float64}, idx::Int)
    return squeeze(sum(mat, idx), idx)
end

"""
Sample data given hyperparameters.
"""
function sample_data(N::Int, model::DRModel)
    D = model.D
    M = model.M
    W = zeros(M, D)
    mu = zeros(D)
    for d in 1 : D
        W[:,d] = rand(MvNormal(model.m_W[:,d], model.Sigma_W[:,:,d]))
    end
    mu = rand(MvNormal(model.m_mu, model.Sigma_mu))
    
    Y = zeros(D, N)
    X = randn(M, N)
    for n in 1 : N
        Y[:,n] = rand(MvNormal(W'*X[:,n] + mu, model.sigma2_y*eye(D)))
    end
    return Y, X, W, mu
end

function init(Y::Array{Float64, 2}, prior::DRModel)
    M = prior.M
    D, N = size(Y)
    X = randn(M, N)
    XX = zeros(M, M, N)
    for n in 1 : N
        XX[:,:,n] = X[:,n]*X[:,n]' + eye(M)
    end
    return X, XX
end

function update_W(Y::Array{Float64, 2}, prior::DRModel, posterior::DRModel,
                  X::Array{Float64, 2}, XX::Array{Float64, 3})
    D = prior.D
    M = prior.M
    N = size(Y, 2)
    m_W = zeros(M, D)
    Sigma_W = zeros(M, M, D)
    mu = posterior.m_mu
    for d in 1 : D
        Sigma_W[:,:,d] = inv(inv(prior.sigma2_y)*sqsum(XX, 3) + inv(prior.Sigma_W[:,:,d]))
        m_W[:,d] = Sigma_W[:,:,d]*(inv(prior.sigma2_y)*X*(Y[[d],:] - mu[d]*ones(1, N))'
                                   + inv(prior.Sigma_W[:,:,d])*prior.m_W[:,d])
    end
    return DRModel(D, M, prior.sigma2_y, m_W, Sigma_W, posterior.m_mu, posterior.Sigma_mu)
end

function update_mu(Y::Array{Float64, 2}, prior::DRModel, posterior::DRModel,
                   X::Array{Float64, 2}, XX::Array{Float64, 3})
    N = size(Y, 2)
    D = prior.D
    M = prior.M
    W = posterior.m_W
    Sigma_mu = inv(N*inv(prior.sigma2_y)*eye(D) + inv(prior.Sigma_mu))
    m_mu = Sigma_mu*(inv(prior.sigma2_y)*sqsum(Y - W'*X, 2) + inv(prior.Sigma_mu)*prior.m_mu)
    return DRModel(D, M, prior.sigma2_y, posterior.m_W, posterior.Sigma_W, m_mu, Sigma_mu)
end

function update_X(Y::Array{Float64, 2}, posterior::DRModel)
    D, N = size(Y)
    M = posterior.M
    
    W = posterior.m_W
    WW = zeros(M, M, D)
    for d in 1 : D
        WW[:,:,d] = W[:,d]*W[:,d]' + posterior.Sigma_W[:,:,d]
    end
    mu = posterior.m_mu
    X = zeros(M, N)
    XX = zeros(M, M, N)
    for n in 1 : N
        Sigma = inv(inv(posterior.sigma2_y)*sqsum(WW, 3) + eye(M))
        X[:,n] = inv(posterior.sigma2_y)*Sigma*W*(Y[:,n] - mu)
        XX[:,:,n] = X[:,n] * X[:,n]' + Sigma
    end
    return X, XX
end

function interpolate(mask::BitArray{2}, X::Array{Float64, 2}, posterior::DRModel)
    Y_est = posterior.m_W'*X + repmat(posterior.m_mu, 1, size(X, 2))
    return return Y_est[mask]
end

"""
Compute variational posterior distributions.
"""
function VI(Y::Array{Float64, 2}, prior::DRModel, max_iter::Int)
    X, XX = init(Y, prior)
    mask = isnan.(Y)
    sum_nan = sum(mask)
    posterior = deepcopy(prior)

    #progress = Progress(max_iter)
    for iter in 1 : max_iter
        # progress
        #next!(progress)
        
        # Interpolate
        if sum_nan > 0
            Y[mask] = interpolate(mask, X, posterior)
        end
        
        # M-step
        posterior = update_W(Y, prior, posterior, X, XX)
        posterior = update_mu(Y, prior, posterior, X, XX)
        
        # E-step
        X, XX = update_X(Y, posterior)
    end
    
    return posterior, X
end

end
