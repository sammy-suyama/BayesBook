###################################
## Demo script for Bayesian Dimensionality Reduction

using PyPlot, PyCall
@pyimport sklearn.datasets as datasets

push!(LOAD_PATH,".")
import DimensionalityReduction

function load_facedata(skip::Int)
    face = datasets.fetch_olivetti_faces()
    Y_raw = face["images"]
    N, S_raw, _ = size(Y_raw)

    L = round(Int, S_raw / skip)
    Y_tmp = Y_raw[:,1:skip:end, 1:skip:end]
    Y = convert(Array{Float64, 2}, reshape(Y_tmp, N, size(Y_tmp,2)*size(Y_tmp,3))')
    D = size(Y, 1)
    
    return Y, D, L
end

function visualize(Y::Array{Float64,2}, L::Int)
    D, N = size(Y)
    base = round(Int, sqrt(N))
    v = round(Int, (L*ceil(N / base)))
    h = L * base
    pic = zeros(v, h)

    for n in 1 : N
        i = round(Int, (L*ceil(n / base)))
        idx1 = i - L + 1 : i
        idx2 = L*mod(n-1, base)+1 : L*(mod(n-1, base) + 1)
        pic[idx1,idx2] = reshape(Y[:,n], L, L)
    end
    imshow(pic, cmap=ColorMap("gray"))
end

function visualize(Y::Array{Float64,2}, L::Int, mask::BitArray{2})
    # for missing
    D, N = size(Y)
    base = round(Int, sqrt(N))
    v = round(Int, (L*ceil(N / base)))
    h = L * base
    pic = zeros(v, h, 3)

    Y_3dim = zeros(D, N, 3)
    for i in 1 : 3
        if i == 2
            Y_tmp = deepcopy(Y)
            Y_tmp[mask] = 1
            Y_3dim[:,:,i] = Y_tmp
        else
            Y_tmp = deepcopy(Y)
            Y_tmp[mask] = 0
            Y_3dim[:,:,i] = Y_tmp
        end
    end
    
    for n in 1 : N
        i = round(Int, (L*ceil(n / base)))
        idx1 = i - L + 1 : i
        idx2 = L*mod(n-1, base)+1 : L*(mod(n-1, base) + 1)
        for i in 1 : 3
            pic[idx1,idx2,i] = reshape(Y_3dim[:,n,i], L, L)
        end
    end
    imshow(pic, cmap=ColorMap("gray"))
end

"""
Run a demo script of missing data interpolation for face dataset.
"""
function test_face_missing()
    # load data
    skip = 2
    Y, D, L = load_facedata(skip)

    # mask
    missing_rate = 0.50
    mask = rand(size(Y)) .< missing_rate
    Y_obs = deepcopy(Y)
    Y_obs[mask] = NaN
    
    # known parames
    M = 16
    sigma2_y = 0.001
    Sigma_W = zeros(M,M,D)
    Sigma_mu = 1.0 * eye(D)
    for d in 1 : D
        Sigma_W[:,:,d] = 0.1 * eye(M)
    end
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, zeros(M, D), Sigma_W, zeros(D), Sigma_mu)

    # learn & generate
    max_iter = 100
    posterior, X_est = DimensionalityReduction.VI(deepcopy(Y_obs), prior, max_iter)
    Y_est = posterior.m_W'*X_est + repmat(posterior.m_mu, 1, size(X_est, 2))
    Y_itp = deepcopy(Y_obs)
    Y_itp[mask] = Y_est[mask]
  
    #visualize
    N_show = 4^2
    
    figure("Observation")
    clf()
    visualize(Y_obs[:,1:N_show], L, mask[:,1:N_show])
    title("Observation")

    #figure("Estimation")
    #clf()
    #visualize(Y_est[:,1:N_show], L)
    #title("Estimation")

    figure("Interpolation")
    clf()
    visualize(Y_itp[:,1:N_show], L)
    title("Interpolation")

    figure("Truth")
    clf()
    visualize(Y[:,1:N_show], L)
    title("Truth")
    show()
end

"""
Run a dimensionality reduction demo using Iris dataset.
"""
function test_iris()
    ##################
    # load data
    iris = datasets.load_iris()
    Y_obs = iris["data"]'
    label_list = [iris["target_names"][elem+1] for elem in iris["target"]]
    D, N = size(Y_obs)

    ##################
    # 2D compression    

    # model
    M = 2
    sigma2_y = 0.001
    Sigma_W = zeros(M,M,D)
    Sigma_mu = 1.0 * eye(D)
    for d in 1 : D
        Sigma_W[:,:,d] = 0.1 * eye(M)
    end
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, zeros(M, D), Sigma_W, zeros(D), Sigma_mu)
        
    # learn & generate
    max_iter = 100
    posterior, X_est = DimensionalityReduction.VI(deepcopy(Y_obs), prior, max_iter)

    # visualize
    figure("2D plot")
    clf()
    scatter(X_est[1,1:50], X_est[2,1:50], color="r")
    scatter(X_est[1,51:100], X_est[2,51:100], color="g")
    scatter(X_est[1,101:end], X_est[2,101:end], color="b")
    xlabel("\$x_1\$", fontsize=20)
    ylabel("\$x_2\$", fontsize=20)
    legend([label_list[1], label_list[51], label_list[101]], fontsize=16)

    ##################
    # 3D compression

    # model
    M = 3
    sigma2_y = 0.001
    Sigma_W = zeros(M,M,D)
    Sigma_mu = 1.0 * eye(D)
    for d in 1 : D
        Sigma_W[:,:,d] = 0.1 * eye(M)
    end
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, zeros(M, D), Sigma_W, zeros(D), Sigma_mu)
        
    # learn & generate
    max_iter = 100
    posterior, X_est = DimensionalityReduction.VI(deepcopy(Y_obs), prior, max_iter)

    # visualize
    figure("3D plot")
    clf()
    scatter3D(X_est[1,1:50], X_est[2,1:50], X_est[3,1:50], c="r")
    scatter3D(X_est[1,51:100], X_est[2,51:100], X_est[3,51:100], c="g")
    scatter3D(X_est[1,101:end], X_est[2,101:end], X_est[3,101:end], c="b")
    legend([label_list[1], label_list[51], label_list[101]], fontsize=16)
    xlabel("\$x_1\$", fontsize=20)
    ylabel("\$x_2\$", fontsize=20)
    zlabel("\$x_3\$", fontsize=20)
    show()
end

#test_face_missing()
test_iris()
