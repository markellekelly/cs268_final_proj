module EVGONN

export NeuralNetwork, predict, train, prepare, backup

sigmoid(x) = 1. / (1 + exp(-x))

function ∇sigmoid(x)
    x = sigmoid(x)
    return x * (1 - x)
end

LeakyReLU(x) = max(x, 0.01x)

∇LeakyReLU(x) = x > 0 ? 1 : 0.01

ELU(x) = max(x, 0.1 * (exp(x) - 1))

∇ELU(x) = max(1, 0.1 * (exp(x)))

function ∇tanh(x)
    x = tanh(x)
    return 1 - x * x
end

mutable struct Layer
    # weights and biases
    all_synapses::Array
    bias::Array
    # activations
    out::Function
    ∇out::Function
    # saved values
    input::Array
    net::Array
    output::Array

    function Layer(ninput::Int, noutput::Int, t; isoutputlayer=false)
        bias = isoutputlayer ? zeros(1, noutput) : randn(1, noutput)
        synapses = randn(ninput, noutput)
        all_synapses = [synapses]
        for i in 1:t-1
            push!(all_synapses, randn(ninput, noutput))
        end
        return new(all_synapses, bias, sigmoid, ∇sigmoid)
    end
end

mutable struct NeuralNetwork
    layers::Array{Layer}
    learning_rate::Float64
    β1::Float64
    β2::Float64
    t::Int64
    best_synapses::Int64

    function NeuralNetwork(ninput::Int64, nhidden::Tuple, noutput::Int64, t::Int64; η=0.01, β1=0.01, β2=0.01)
        l = [Layer(ninput, nhidden[1],t)]
        for i in 1:(length(nhidden) - 1)
            push!(l, Layer(nhidden[i], nhidden[i+1]))
        end
        push!(l, Layer(nhidden[end], noutput, t, isoutputlayer=true))
        return new(l, η, β1, β2, t)
    end

    function NeuralNetwork(dumpdata)
        return new(dumpdata["layers"], dumpdata["lr"])
    end
end


function feedforward(data::Array, nn::NeuralNetwork, s::Int64)
    for layer in nn.layers
        layer.input = data
        data = data * layer.all_synapses[s] .+ layer.bias
        layer.net = data
        data = layer.out.(data)
        layer.output = data
    end
    return data
end

err(prediction, target) = 0.5 * (target - prediction) ^ 2

∇err(prediction, target) = prediction - target

function backpropagate(output::Array, nn::NeuralNetwork, s::Int64, best::Bool)
    η = nn.learning_rate
    layer = nn.layers[end]
    # calculate partial derivatives
    ∂err_∂out = ∇err.(layer.output, output)
    ∂out_∂net = layer.∇out.(layer.net)
    δ = ∂err_∂out .* ∂out_∂net
    # update weights
    layer.all_synapses[s] += -η * layer.input' * δ
    # hidden layers
    for i in length(nn.layers)-1:-1:1
        layer = nn.layers[i]
        ∂err_∂out = δ * transpose(nn.layers[i + 1].all_synapses[s])
        ∂out_∂net = layer.∇out.(layer.net)
        δ = ∂err_∂out .* ∂out_∂net
        if best
            layer.all_synapses[s] += -η * transpose(layer.input) * δ
        else
            # todo: replace 0 with the Lf thing
            layer.all_synapses[s] += -nn.β1 * transpose(layer.input) * δ - nn.β2 * norm(Lf(s))
        end
        layer.bias = layer.bias .+ (-η * δ)
    end
end

function prepare(data)::Array{Float64, 2}
    data = cat(2, data)
    return data
end

function Lf(s)
    current = nn.all_synapses[s]
    best = nn.best_synapses
    x_diff = xi - nn.all_synapses[best] + 0.01
    res1 = feedforward(data, nn, s)
    cost1 = err.(res1, output)
    cost1 = sum(cost1)
    res2 = feedforward(data, nn, best)
    cost2 = err.(res2, output)
    cost2 = sum(cost2)
    cost_diff = cost1 - cost2
    return cost_diff/x_diff
end

function predict(data::Array, output::Array, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn, nn.best_synapses)
    # calculate cost
    cost = err.(result, output)
    cost = sum(cost)
    return Dict("result" => result, "cost" => cost)
end

function train(data::Array, output::Array, nn::NeuralNetwork)
    best_cost_ind = 1
    result = feedforward(data, nn, 1)
    best_cost = sum(err.(result, output))
    for s in 2:nn.t
        # propagate forward
        result = feedforward(data, nn, s)
        # calculate cost
        cost = err.(result, output)
        cost = sum(cost)
        if cost < best_cost
            best_cost_ind = s
            best_cost = cost
        end
    end
    for s in 1:nn.t
        backpropagate(output, nn, s, s == best_cost_ind)
    end
    nn.best_synapses = best_cost_ind
    return Dict("result" => result, "cost" => cost)
end

function predict(data::Array, nn::NeuralNetwork)
    result = feedforward(data, nn)
    return Dict("result" => result)
end

function backup(nn)
    return Dict("layers" => nn.layers, "lr" => nn.learning_rate)
end

end