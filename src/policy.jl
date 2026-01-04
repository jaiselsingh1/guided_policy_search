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
# this makes the struct NeuralPolicy callable like an object while functional 
# this is useful given that the Struct needs to have it's params/weights 
# overloading the () function 
function(policy::NeuralPolicy)(state::AbstractVector, params)
    return policy.chain(state, params)
end

# initialize the parameters 
function init_params(policy::NeuralPolicy; rng::AbstractRNG = Random.GLOBAL_RNG)
    return SimpleChains.init_params(policy.chain; rng = rng)
end 

"""train_policy!(policy, states, actions, params; epochs=50, learning_rate=1e-3)
train the policy with SL to imitate the (state, action) pairs

return the final training loss
"""

function train_policy!(
    policy::NeuralPolicy, 
    states::Matrix{Float64},
    actions::Matrix{Float64},
    params;  # the colon seperates out the positional and keyword arguments from one another
    epochs::Int = 50, 
    learning_rate::Float64 = 1e-3, 
    batch_size::Int = 256
)
    N = size(states, 2)  # number of training samples

    # SimpleChains expects F32 
    states_f32 = Float32.(states)
    actions_f32 = Float32.(actions)

    # loss function arch 
    function loss_fn(p)
        total_loss = 0.0f0
        for i in 1:N 
            pred = policy(states_f32[:, i], p)
            target = actions_f32[:, i]
            total_loss += sum((pred .- target).^2)
        end 
        return total_loss / N
    end 

    # training loop with simple gradient descent 
    for epoch in 1:num_epochs
        # compute the loss and the gradient 
        current_loss = loss_fn(params)

        # compute the gradient using FD 
        # allocate gradient storage
        grad = SimpleChains.alloc_threaded_grad(policy.chain)

        SimpleChains.grad!(grad, policy.chain, loss_fn, params, states_f32, actions_f32)

        # gradient descent update 
        params .-= learning_rate .* grad 

        if epoch % 10 == 0
        println("epoch $epoch / $epochs, loss = $(current_loss)")
        end 
    end 

    final_loss = loss_fn(params)
    return final_loss
end 

function save_policy(
    filename::String, 
    policy::NeuralPolicy, 
    params
)
    JLD2.@save filename policy params 
    println("saved policy to $filename")
end 

function load_policy(filename::String)
    JLD2.@load filename policy params
    println("loaded policy from $filename")
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


