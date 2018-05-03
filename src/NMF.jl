"""
Variational inference for Bayesian NMF
"""
module NMF
using Distributions
using StatsFuns.logsumexp, SpecialFunctions.digamma

export NMFModel
export sample_data, VI

####################
## Types
struct NMFModel
    a_t::Array{Float64, 2} # D x K
    b_t::Array{Float64, 2} # D x L
    a_v::Float64 # 1 dim
    b_v::Float64 # 1 dim
end

function sqsum(mat::Array, idx)
    return squeeze(sum(mat, idx), idx)
end

####################
## functions
function init(X::Array{Int64, 2}, model::NMFModel)
    D, N = size(X)
    K = size(model.a_t, 2)
    S = zeros(D, K, N)
    A_t = rand(D, K)
    B_t = rand(D, K)
    A_v = rand(K, N)
    B_v = rand(K, N)
    for d in 1 : D
        for k in 1 : K
            for n in 1 : N
                S[d,k,n] = X[d,n] * A_t[d,k] * B_t[d,k] * A_v[k,n] * B_v[k,n]
            end
        end
    end   
    return S, A_t, B_t, A_v, B_v
end

function update_S(X::Array{Int64, 2}, A_t::Array{Float64, 2}, B_t::Array{Float64, 2}, A_v::Array{Float64, 2}, B_v::Array{Float64, 2})
    D, K = size(A_t)
    N = size(A_v, 2)
    S = zeros(D, K, N)
    for d in 1 : D
        for n in 1 : N
            # K dim
            ln_P = (digamma.(A_t[d,:]) + log.(B_t[d,:])
                    + digamma.(A_v[:,n]) + log.(B_v[:,n])
                    )
            ln_P = ln_P - logsumexp(ln_P)
            S[d,:,n] = X[d,n] * exp.(ln_P)
        end
    end
    return S
end

function update_T(S::Array{Float64, 3}, A_v::Array{Float64, 2}, B_v::Array{Float64, 2}, model::NMFModel)
    D, K, N = size(S)
    a_t = model.a_t # DxK
    b_t = model.b_t # DxK
    A_t = a_t + sqsum(S, 3)
    B_t = (a_t ./ b_t + repmat(sqsum(A_v.*B_v, 2)', D, 1)).^(-1)
    return A_t, B_t
end

function update_V(S::Array{Float64, 3}, A_t::Array{Float64, 2}, B_t::Array{Float64, 2}, model::NMFModel)
    a_v = model.a_v
    b_v = model.b_v
    D, K, N = size(S)
    A_v = a_v + sqsum(S, 1)
    B_v = (a_v / b_v + repmat(sqsum(A_t.*B_t, 1), 1, N)).^(-1)
    return A_v, B_v
end

"""
Sample data given hyperparameters.
"""
function sample_data(N::Int, model::NMFModel)
    # TODO; check b or 1/b ?
    D, K = size(model.a_t)

    T = zeros(D, K)
    for d in 1 : D
        for k in 1 : K
            T[d,k] = rand(Gamma(model.a_t[d,k], 1.0/model.b_t[d,k])) # TODO: check
        end
    end

    V = reshape(rand(Gamma(model.a_v, 1.0/model.b_v), K*N) , K, N) # TODO: check
    
    S = zeros(D, K, N)
    for d in 1 : D
        for k in 1 : K
            for n in 1 : N
                S[d,k,n] = T[d,k] * V[k,n]
            end
        end
    end
    #X = sqsum(S, 2) + 0.0 # zero noise
    X = sqsum(S, 2)
    return X, T, S, V
end

function update_model(A_t::Array{Float64, 2}, B_t::Array{Float64, 2}, model::NMFModel)
    return NMFModel(A_t, B_t, model.a_v, model.b_v)
end

"""
Compute variational posterior distributions.
"""
function VI(X::Array{Int64, 2}, model::NMFModel, max_iter::Int)
    K = size(model.a_t, 2)
    D, N = size(X)
    S, A_t, B_t, A_v, B_v = init(X, model)
    for iter in 1 : max_iter
        # latent
        S = update_S(X, A_t, B_t, A_v, B_v)
        A_v, B_v = update_V(S, A_t, B_t, model)

        # param
        A_t, B_t = update_T(S, A_v, B_v, model)
    end
    
    return update_model(A_t, B_t, model), S, A_t.*B_t, A_v.*B_v
end

end
