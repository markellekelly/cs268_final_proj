import Pkg
include("EVGONN.jl")
using .EVGONN
using StatsBase
using MLDatasets
using DelimitedFiles

# load and prep data
n_categories = 10
n_var = 784
n_train = 60000
n_test = 10000
n = 70000
train_x, train_y = FashionMNIST.traindata()
test_x, test_y  = FashionMNIST.testdata()
X_train = reshape(convert(Array{Float64}, train_x), (n_var, n_train))
X_test = reshape(convert(Array{Float64}, test_x), (n_var, n_test))
X = hcat(X_train, X_test)
dt = fit(ZScoreTransform, Array(X), dims=1)
X = StatsBase.transform(dt, X);
X_train = reshape(X[:,1:n_train], (n_var, 1, n_train));
X_test = reshape(X[:,(n_train+1):n], (n_var, 1, n_test));
train_y = train_y.+ 1;
y_test = test_y.+ 1;
y_train = zeros(UInt8, (n_train, n_categories))
for i in 1:n_train
    y_train[i, train_y[i]] = 0x01
end

# define funcs
function check(nn)
    checks = Dict("HIT" => 0, "MISS" => 0)
    for i in 1:n_test
        if argmax(EVGONN.predict(EVGONN.prepare(X_test[:, 1, i]'), nn)["result"])[2] == y_test[i]
            checks["HIT"] += 1
        else
            checks["MISS"] += 1
        end
    end
    return checks["HIT"] / n_test
end

function training(nn, k, iters=10000)
    start = time()
    costs = zeros(n_train)
    old_costs = sum(costs)
    losses = zeros(2000)
    j = 0
    for it in 1:iters
        for i in 1:n_train
            result = EVGONN.train(EVGONN.prepare(X_train[:, 1, i]'), EVGONN.prepare(y_train[i, :]'), nn)
            costs[i] = result["cost"]
        end
        if it % 5 == 0
            new_costs = sum(costs)
            losses[j] = new_costs
            fname = string("losses-", k, ".txt")
            writedlm(fname, losses)
            j = j + 1
            if abs(new_costs - old_costs) < 25
                total_time = (time() - start)
                test_acc = check(nn)
                return [k, new_costs, test_acc, it, total_time]
            end
        end
        if it % 10 == 0
            nn.learning_rate = nn.learning_rate * 0.9
            nn.β1 = nn.β1 * 0.95
        end
    end
    new_costs = sum(costs)
    total_time = (time() - start)
    test_acc = check(nn)
    return [k, new_costs, test_acc, iters, total_time]
end

# experiment: train for different values of k
ks = [2,3,4,5]
res = []
for k in ks
    nn = EVGONN.NeuralNetwork(n_var, (40,20,15), n_categories, k, η=0.05, β1=0.02, β2=0.0000001);
    push!(res, training(nn, k))
    writedlm("out.txt", res)
end