using SimpleChains 
using Statistics 
using Random 
using JLD2 

"""Neural Policy 
- feedforward neural network that maps states to actions 
chain:: SimpleChains neural network 
state_dim:: dimension of the state space 
action_dim:: dimension of the action space 
"""

struct NeuralPolicy 
    chain::SimpleChain 
    state_dim::Int 
    action_dim::Int 
end 

function NeuralPolicy(state_dim::Int, action_dim::Int, hidden_sizes::Vector{Int} = [64, 64])
    # build the network layer by layer 
    layers = Any[]

    # turbo dense uses SIMD optimization 
    push!(layers, TurboDense(tanh, state_dim, hidden_sizes[1]))
    for i in 1:length(hidden_sizes)-1
        push!(layers, TurboDense(tanh, hidden_sizes[i], hidden_sizes[i+1]))
    end 

    # output layer 
    push!(layers, TurboDense(tanh, hidden_sizes[end], action_dim))

    # the ... does unpacking since SimpleChain wants to have each layer as an input 
    chain = SimpleChain(Tuple(layers)...)
    return NeuralPolicy(chain, state_dim, action_dim)

end 

"""
(policy::NeuralPolicy)(state, params)
forward pass: compute the action from the state 
"""
# this makes the function NeuralPolicy callable like an object 

function(policy::NeuralPolicy)(state::AbstractVector, params)
    return policy.chain(state, params)
end 

function describe_policy(policy::NeuralPolicy)
    println("NeuralPolicy Structure:")
    println("  - Input Dimension:  $(policy.state_dim)")
    println("  - Output Dimension: $(policy.action_dim)")
    println("---")
    display(policy.chain)
end

p = NeuralPolicy(4, 2, [32, 32])
describe_policy(p)


