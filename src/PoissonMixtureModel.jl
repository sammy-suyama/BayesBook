"""
Bayesian Poisson Mixture Model
"""
module PoissonMixtureModel
using StatsFuns.logsumexp, SpecialFunctions.digamma
using Distributions

export Gam, BPMM, Poi, PMM
export sample_PMM, sample_data, winner_takes_all
export learn_GS, learn_CGS, learn_VI

####################
## Types
struct Gam
    # Parameters of Gamma distribution
    a::Vector{Float64}
    b::Float64
end

struct BPMM
    # Parameters of Bayesian Poisson Mixture Model 
    D::Int
    K::Int
    alpha::Vector{Float64}
    cmp::Vector{Gam}
end

struct Poi
    # Parameters of Poisson Distribution
    lambda::Vector{Float64}
end

struct PMM
    # Parameters of Poisson Mixture Model
    D::Int
    K::Int
    phi::Vector{Float64}
    cmp::Vector{Poi}
end

####################
## Common functions
"""
Sample a PMM given hyperparameters.
"""
function sample_PMM(bpmm::BPMM)
    cmp = Vector{Poi}()
    for c in bpmm.cmp
        lambda = Vector{Float64}()
        for d in 1 : bpmm.D
            push!(lambda, rand(Gamma(c.a[d], 1.0/c.b)))
        end
        push!(cmp, Poi(lambda))  
    end
    phi = rand(Dirichlet(bpmm.alpha))
    return PMM(bpmm.D, bpmm.K, phi, cmp)
end

"""
Sample data from a specific PMM model.
"""
function sample_data(pmm::PMM, N::Int)
    X = zeros(pmm.D, N)
    S = categorical_sample(pmm.phi, N)
    for n in 1 : N
        k = indmax(S[:, n])
        for d in 1 : pmm.D
            X[d,n] = rand(Poisson(pmm.cmp[k].lambda[d]))
        end
    end
    return X, S
end

categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:,1]
function categorical_sample(p::Vector{Float64}, N::Int)
    K = length(p)
    S = zeros(K, N)
    S_tmp = rand(Categorical(p), N)
    for k in 1 : K
        S[k,find(S_tmp.==k)] = 1
    end
    return S
end

function init_S(X::Matrix{Float64}, bpmm::BPMM)
    N = size(X, 2)
    K = bpmm.K
    S = categorical_sample(ones(K)/K, N)    
    return S
end

function calc_ELBO(X::Matrix{Float64}, pri::BPMM, pos::BPMM)
    ln_expt_S = update_S(pos, X)
    expt_S = exp.(ln_expt_S)
    K, N = size(expt_S)
    D = size(X, 1)

    expt_ln_lambda = zeros(D, K)
    expt_lambda = zeros(D, K)
    expt_ln_lkh = 0
    for k in 1 : K
        expt_ln_lambda[:,k] = digamma.(pos.cmp[k].a) - log.(pos.cmp[k].b)
        expt_lambda[:,k] = pos.cmp[k].a / pos.cmp[k].b
        for n in 1 : N
            expt_ln_lkh += expt_S[k,n] * (X[:, n]' * expt_ln_lambda[:,k]
                                       - sum(expt_lambda[:,k]) - sum(lgamma.(X[:,n]+1)))[1]
        end
    end
    
    expt_ln_pS = sum(expt_S' * (digamma.(pos.alpha) - digamma.(sum(pos.alpha))))
    expt_ln_qS = sum(expt_S .* ln_expt_S)
    
    KL_lambda = 0
    for k in 1 : K
        KL_lambda += (sum(pos.cmp[k].a)*log.(pos.cmp[k].b) - sum(pri.cmp[k].a)*log.(pri.cmp[k].b)
                      - sum(lgamma.(pos.cmp[k].a)) + sum(lgamma.(pri.cmp[k].a))
                      + (pos.cmp[k].a - pri.cmp[k].a)' * expt_ln_lambda[:,k]
                      + (pri.cmp[k].b - pos.cmp[k].b) * sum(expt_lambda[:,k])
                      )[1]
    end
    KL_pi = (lgamma.(sum(pos.alpha)) - lgamma.(sum(pri.alpha))
             - sum(lgamma.(pos.alpha)) + sum(lgamma.(pri.alpha))
             + (pos.alpha - pri.alpha)' * (digamma.(pos.alpha) - digamma.(sum(pos.alpha)))
             )[1]
    
    VB = expt_ln_lkh + expt_ln_pS - expt_ln_qS - (KL_lambda + KL_pi)
    return VB
end

function add_stats(bpmm::BPMM, X::Matrix{Float64}, S::Matrix{Float64})
    D = bpmm.D
    K = bpmm.K
    sum_S = sum(S, 2)
    alpha = [bpmm.alpha[k] + sum_S[k] for k in 1 : K]
    cmp = Vector{Gam}()

    XS = X*S';
    for k in 1 : K
        a = [(bpmm.cmp[k].a[d] + XS[d,k])::Float64 for d in 1 : D]
        b = bpmm.cmp[k].b + sum_S[k]
        push!(cmp, Gam(a, b))
    end
    return BPMM(D, K, alpha, cmp)
end

remove_stats(bpmm::BPMM, X::Matrix{Float64}, S::Matrix{Float64}) = add_stats(bpmm, X, -S)

####################
## used for Variational Inference
function update_S(bpmm::BPMM, X::Matrix{Float64})
    D, N = size(X)
    K = bpmm.K
    ln_expt_S = zeros(K, N)
    tmp = zeros(K)

    sum_digamma_tmp = digamma.(sum(bpmm.alpha))
    for k in 1 : K
        tmp[k] = - sum(bpmm.cmp[k].a) / bpmm.cmp[k].b
        tmp[k] += digamma.(bpmm.alpha[k]) - sum_digamma_tmp
    end
    ln_lambda_X = [X'*(digamma.(bpmm.cmp[k].a) - log.(bpmm.cmp[k].b)) for k in 1 : K]
    for n in 1 : N
        tmp_ln_pi =  [tmp[k] + ln_lambda_X[k][n] for k in 1 : K]
        ln_expt_S[:,n] = tmp_ln_pi - logsumexp(tmp_ln_pi)
    end
    return ln_expt_S
end

"""
Pick single states having a max probability.
"""
function winner_takes_all(S::Matrix{Float64})
    S_ret = zeros(size(S))
    for n in 1 : size(S_ret, 2)
        idx = indmax(S[:,n])
        S_ret[idx,n] = 1
    end
    return S_ret
end

####################
## used for Gibbs Sampling
function sample_S_GS(pmm::PMM, X::Matrix{Float64})
    D, N = size(X)
    K = pmm.K
    S = zeros(K, N)

    tmp = [-sum(pmm.cmp[k].lambda) + log.(pmm.phi[k]) for k in 1 : K]
    ln_lambda_X = [X'*log.(pmm.cmp[k].lambda) for k in 1 : K]
    for n in 1 : N
        tmp_ln_phi = [(tmp[k] + ln_lambda_X[k][n])::Float64 for k in 1 : K]
        tmp_ln_phi = tmp_ln_phi - logsumexp(tmp_ln_phi)
        S[:,n] = categorical_sample(exp.(tmp_ln_phi))
    end
    return S
end

####################
## used for Collapsed Gibbs Sampling
function calc_ln_NB(Xn::Vector{Float64}, gam::Gam)
    ln_lkh = [(gam.a[d]*log.(gam.b)
               - lgamma.(gam.a[d])
               + lgamma.(Xn[d] + gam.a[d])
               - (Xn[d] + gam.a[d])*log.(gam.b + 1)
               )::Float64 for d in 1 : size(Xn, 1)]
    return sum(ln_lkh)
end

function sample_Sn(Xn::Vector{Float64}, bpmm::BPMM)
    ln_tmp = [(calc_ln_NB(Xn, bpmm.cmp[k]) + log.(bpmm.alpha[k])) for k in 1 : bpmm.K]
    ln_tmp = ln_tmp -  logsumexp(ln_tmp)
    Sn = categorical_sample(exp.(ln_tmp))
    return Sn
end

function sample_S_CGS(S::Matrix{Float64}, X::Matrix{Float64}, bpmm::BPMM)
    D, N = size(X)
    K = size(S, 1)
    for n in randperm(N)
        # remove
        bpmm = remove_stats(bpmm, X[:,[n]], S[:,[n]])
        # sample
        S[:,n] = sample_Sn(X[:,n], bpmm)
        # insert
        bpmm = add_stats(bpmm, X[:,[n]], S[:,[n]])
    end
    return S, bpmm
end

####################
## Algorithm main
"""
Compute posterior distribution via variational inference.
"""
function learn_VI(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    # initialisation
    expt_S = init_S(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, expt_S)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # E-step
        expt_S = exp.(update_S(bpmm, X))
        # M-step
        bpmm = add_stats(prior_bpmm, X, expt_S)
        # calc VB
        VB[i] = calc_ELBO(X, prior_bpmm, bpmm)
    end

    return expt_S, bpmm, VB
end

"""
Compute posterior distribution via Gibbs sampling.
"""
function learn_GS(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    # initialisation
    S = init_S(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, S)
    VB = NaN * zeros(max_iter)
    
    # inference
    for i in 1 : max_iter            
        # sample parameters
        pmm = sample_PMM(bpmm)
        # sample latent variables
        S = sample_S_GS(pmm, X)
        # update current model
        bpmm = add_stats(prior_bpmm, X, S)
        # calc VB
        VB[i] = calc_ELBO(X, prior_bpmm, bpmm)
    end

    return S, bpmm, VB
end

"""
Compute posterior distribution via collapsed Gibbs sampling.
"""
function learn_CGS(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    # initialisation
    S = init_S(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, S)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # directly sample S
        S, bpmm = sample_S_CGS(S, X, bpmm)
        # calc VB
        VB[i] = calc_ELBO(X, prior_bpmm, bpmm)
    end

    return S, bpmm, VB
end

end
