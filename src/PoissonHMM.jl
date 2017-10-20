"""
Bayesian 1dim Poisson Hidden Markov Model
"""
module PoissonHMM
using StatsFuns.logsumexp, SpecialFunctions.digamma
using Distributions

export Gam, GHMM, Poi, HMM
export sample_HMM, sample_data, winner_takes_all
export learn_VI

####################
## Types
struct Gam
    # Parameters of Gamma distribution
    # 1dim
    a::Float64
    b::Float64
end

struct BHMM
    # Parameters of Bayesian Bernoulli Mixture Model 
    K::Int
    alpha_phi::Vector{Float64}
    alpha_A::Matrix{Float64}    
    cmp::Vector{Gam}
end

struct Poi
    # Parameters of Poisson Distribution
    # 1 dim
    lambda::Float64
end

struct HMM
    # Parameters of Bernoulli Mixture Model
    K::Int
    phi::Vector{Float64}
    A::Matrix{Float64}
    cmp::Vector{Poi}
end

####################
## Common functions
"""
Sample an HMM from prior
"""
function sample_HMM(bhmm::BHMM)
    cmp = Vector{Poi}()
    for c in bhmm.cmp
        lambda = rand(Gamma(c.a, 1.0/c.b))
        push!(cmp, Poi(lambda))
    end
    phi = rand(Dirichlet(bhmm.alpha_phi))
    A = zeros(size(bhmm.alpha_A))
    for k in 1 : bhmm.K
        A[:,k] = rand(Dirichlet(bhmm.alpha_A[:,k]))
    end
    return HMM(bhmm.K, phi, A, cmp)
end

"""
Sample data from a specific Poisson HMM
"""
function sample_data(hmm::HMM, N::Int)
    X = zeros(N)
    Z = zeros(hmm.K, N)

    # sample (n=1)
    Z[:,1] = categorical_sample(hmm.phi)
    k = indmax(Z[:, 1])
    X[1] = rand(Poisson(hmm.cmp[k].lambda))

    # sample (n>1)
    for n in 2 : N
        Z[:,n] = categorical_sample(hmm.A[:,k])
        k = indmax(Z[:, n])
        X[n] = rand(Poisson(hmm.cmp[k].lambda))
    end
    return X, Z
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

function init_Z(X::Vector{Float64}, bhmm::BHMM)
    N = size(X, 1)
    K = bhmm.K
    
    Z = rand(Dirichlet(ones(K)/K), N)
    ZZ = [zeros(K,K) for _ in 1 : N - 1]
    for n in 1 : N - 1
        ZZ[n] = Z[:,n+1] * Z[:,n]'
    end
    
    return Z, ZZ
end

"""
Not implemented yet.
"""
function calc_ELBO(X::Matrix{Float64}, pri::BHMM, pos::BHMM)
end

function add_stats(bhmm::BHMM, X::Vector{Float64},
                   Z::Matrix{Float64}, ZZ::Vector{Matrix{Float64}})
    K = bhmm.K
    sum_Z = sum(Z, 2)
    alpha_phi = [bhmm.alpha_phi[k] + Z[k,1] for k in 1 : K]
    alpha_A = bhmm.alpha_A + sum(ZZ)
    cmp = Vector{Gam}()
    
    ZX = Z*X # (KxN) x (Nx1) = Kx1
    for k in 1 : K
        a = bhmm.cmp[k].a + ZX[k]
        b = bhmm.cmp[k].b + sum_Z[k]
        push!(cmp, Gam(a, b))
    end
    return BHMM(K, alpha_phi, alpha_A, cmp)
end

remove_stats(bhmm::BHMM, X::Vector{Float64}, Z::Matrix{Float64}) = add_stats(bhmm, X, -Z)

####################
## used for Variational Inference
function update_Z(bhmm::BHMM, X::Vector{Float64}, Z::Matrix{Float64})
    N = size(X, 1)
    K = bhmm.K
    ln_expt_Z = zeros(K, N)

    ln_lkh = zeros(K, N)
    for k in 1 : K
        ln_lambda = digamma.(bhmm.cmp[k].a) - log.(bhmm.cmp[k].b)
        lambda = bhmm.cmp[k].a / bhmm.cmp[k].b
        for n in 1 : N
            ln_lkh[k,n] = X[n]'*(ln_lambda) - lambda
        end
    end
    
    expt_ln_A = zeros(size(bhmm.alpha_A))
    for k in 1 : K
        expt_ln_A[:,k] = digamma.(bhmm.alpha_A[:,k]) - digamma.(sum(bhmm.alpha_A[:,k]))
    end

    # copy
    ln_expt_Z = log.(Z)
    
    # n = 1
    ln_expt_Z[:,1] = (digamma.(bhmm.alpha_phi) - digamma.(sum(bhmm.alpha_phi))
                      + expt_ln_A' * exp.(ln_expt_Z[:,2])
                      + ln_lkh[:,1]
                      )
    ln_expt_Z[:,1] = ln_expt_Z[:,1] - logsumexp(ln_expt_Z[:,1])
    
    # 2 <= n <= N - 1
    for n in 2 : N - 1
        ln_expt_Z[:,n] =( expt_ln_A * exp.(ln_expt_Z[:,n-1])
                          + expt_ln_A' * exp.(ln_expt_Z[:,n+1])
                          + ln_lkh[:,n]
                          )
        ln_expt_Z[:,n] = ln_expt_Z[:,n] - logsumexp(ln_expt_Z[:,n])
    end
    
    # n = N
    ln_expt_Z[:,N] =( expt_ln_A * exp.(ln_expt_Z[:,N-1])
                      + ln_lkh[:,N]
                      )
    ln_expt_Z[:,N] = ln_expt_Z[:,N] - logsumexp(ln_expt_Z[:,N])
    
    # calc output
    Z_ret = exp.(ln_expt_Z)
    ZZ_ret = [zeros(K,K) for _ in 1 : N - 1]
    for n in 1 : N - 1
        ZZ_ret[n] = Z_ret[:,n+1] * Z_ret[:,n]'
    end
    return Z_ret, ZZ_ret
end

"""
Pick single states having a max probability.
"""
function winner_takes_all(Z::Matrix{Float64})
    Z_ret = zeros(size(Z))
    for n in 1 : size(Z_ret, 2)
        idx = indmax(Z[:,n])
        Z_ret[idx,n] = 1
    end
    return Z_ret
end

function logmatprod(ln_A::Array{Float64}, ln_B::Array{Float64})
    I = size(ln_A, 1)
    J = size(ln_B, 2)
    ln_C = zeros(I, J)
    for i in 1 : I
        for j in 1 : J
            ln_C[i, j] = logsumexp(ln_A[i, :] + ln_B[:, j])
        end
    end
    return ln_C
end

function update_Z_fb(bhmm::BHMM, X::Vector{Float64})
    K = bhmm.K
    N = length(X)

    # calc likelihood
    ln_lik = zeros(K, N)
    for k in 1 : K
        ln_lambda = digamma.(bhmm.cmp[k].a) - log.(bhmm.cmp[k].b)
        lambda = bhmm.cmp[k].a / bhmm.cmp[k].b
        for n in 1 : N
            ln_lik[k,n] =X[n]'*(ln_lambda) - lambda                
        end
    end
    expt_ln_phi = digamma.(bhmm.alpha_phi) - digamma.(sum(bhmm.alpha_phi))
    expt_ln_A = zeros(K,K)
    for k in 1 : K
        expt_ln_A[:,k] = digamma.(bhmm.alpha_A[:,k]) - digamma.(sum(bhmm.alpha_A[:,k]))
    end

    Z, ZZ = fb_alg(ln_lik, expt_ln_phi, expt_ln_A)
    
    # different notation
    ZZ_ret = [ZZ[:,:,n] for n in 1:size(ZZ, 3)]

    return Z, ZZ_ret
end

function fb_alg(ln_lik::Matrix{Float64}, ln_phi::Vector{Float64}, ln_A::Matrix{Float64})
    K, T = size(ln_lik)
    ln_Z = zeros(K, T)
    ln_ZZ = zeros(K, K, T)
    ln_alpha = zeros(K, T)
    ln_beta = zeros(K, T)
    ln_st = zeros(T)
    
    for t in 1 : T
        if t == 1
            ln_alpha[:, 1] = ln_phi + ln_lik[:, 1]
        else
            ln_alpha[:, t] = logmatprod(ln_A, ln_alpha[:, t-1]) + ln_lik[:, t]
        end
        ln_st[t] = logsumexp(ln_alpha[:, t])
        ln_alpha[:,t] = ln_alpha[:,t] - ln_st[t]
    end
    
    for t in T-1 : -1 : 1
        ln_beta[:, t] = logmatprod(ln_A', ln_beta[:, t+1] + ln_lik[:,t+1])
        ln_beta[:, t] = ln_beta[:, t] - ln_st[t+1]
    end
    
    ln_Z = ln_alpha + ln_beta
    for t in 1 : T
        if t < T
            ln_ZZ[:,:,t] = (repmat(ln_alpha[:, t]', K, 1) + ln_A
            + repmat(ln_lik[:, t+1] + ln_beta[:,t+1], 1, K))
            ln_ZZ[:,:,t] = ln_ZZ[:,:,t] - ln_st[t+1]
        end
    end
    return exp.(ln_Z), exp.(ln_ZZ)
end

"""
Compute approximate posterior distributions via variational inference.
"""
function learn_VI(X::Vector{Float64}, prior_bhmm::BHMM, max_iter::Int)
    # initialisation
    expt_Z, expt_ZZ = init_Z(X, prior_bhmm)
    bhmm = add_stats(prior_bhmm, X, expt_Z, expt_ZZ)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # E-step
        #expt_Z, expt_ZZ = update_Z(bhmm, X, expt_Z)
        expt_Z, expt_ZZ = update_Z_fb(bhmm, X)
        # M-step
        bhmm = add_stats(prior_bhmm, X, expt_Z, expt_ZZ)
    end

    return expt_Z, bhmm
end
end
