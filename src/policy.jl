using SimpleChains 
using Statistics 
using Random 
using JLD2 
using Zygote

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
    # SimpleChains needs static types hence hardcode for 2 hidden layers
    # turbo dense uses SIMD optimization
    # TurboDense(activation, output_dim) - input dim inferred from previous layer
    if length(hidden_sizes) == 2
        # output layer
        # the ... does unpacking since SimpleChain wants to have each layer as an input
        # static(state_dim) sets the input dimension
        chain = SimpleChain(
            static(state_dim),
            TurboDense(tanh, hidden_sizes[1]),
            TurboDense(tanh, hidden_sizes[2]),
            TurboDense(tanh, action_dim)
        )
    else
        error("Currently only supports 2 hidden layers")
    end
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

    # allocate gradient buffer
    grad_buf = SimpleChains.alloc_threaded_grad(policy.chain)

    # training loop using SimpleChains native approach
    for epoch in 1:epochs
        total_loss = 0.0f0
        fill!(grad_buf, 0.0f0)

        # accumulate loss and gradients
        for i in 1:N
            x_i = states_f32[:, i]
            y_target = actions_f32[:, i]

            # forward pass
            y_pred = policy.chain(x_i, params)

            # compute loss
            loss_val = sum((y_pred .- y_target).^2)
            total_loss += loss_val

            # backward pass using SimpleChains
            # Create loss function that takes output and returns scalar
            target_copy = copy(y_target)
            loss_fn = (out,) -> sum((out .- target_copy).^2)

            # Compute gradient
            SimpleChains.valgrad!(grad_buf, loss_fn, policy.chain, x_i, params)
        end

        # average and update
        current_loss = total_loss / N
        grad_buf ./= N
        params .-= learning_rate .* grad_buf

        if epoch % 10 == 0
            println("epoch $epoch / $epochs, loss = $(current_loss)")
        end
    end

    # compute final loss
    final_loss = 0.0f0
    for i in 1:N
        pred = policy(states_f32[:, i], params)
        target = actions_f32[:, i]
        final_loss += sum((pred .- target).^2)
    end
    final_loss /= N
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
    return policy, params
end 

function describe_policy(policy::NeuralPolicy)
    println("NeuralPolicy Structure:")
    println("  - Input Dimension:  $(policy.state_dim)")
    println("  - Output Dimension: $(policy.action_dim)")
    println("---")
    display(policy.chain)
end

# p = NeuralPolicy(4, 2, [32, 32])
# describe_policy(p)

function rollout_policy(
    env, 
    policy::NeuralPolicy,
    params;
    episode_length::Int = 100, 
    initial_state = nothing)

    model = env.model
    data = env.data

    # reset the environment 
    reset!(model, data)

    if initial_state !== nothing
        nq = length(data.qpos)
        data.qpos .= initial_state[1 : nq]
        data.qvel .= initial_state[nq+1 : end]
    else
        # random perturbation matching MPPI training distribution
        data.qpos .+= 0.2 * randn(length(data.qpos))
        data.qvel .+= 0.2 * randn(length(data.qvel))
    end

    x0 = get_physics_state(model, data)

    # storage 
    states = zeros(env.state_dim, episode_length)
    actions = zeros(env.action_dim, episode_length)
    costs = zeros(episode_length)

    # the rollout loop
    for t in 1:episode_length
        state = get_physics_state(model, data)
        states[:, t] = state

        # forward pass for the policy
        action = policy(state, params)
        actions[:, t] = action

        # apply the action
        data.ctrl .= clamp.(action, -1.0, 1.0)
        step!(model, data)

        # compute the cost
        costs[t] = running_cost_cartpole(data)
    end 

    costs[end] += terminal_cost_cartpole(data)
    return Trajectory(states, actions, costs, x0)
end 

function train_policy_from_mppi(
    env; 
    data_file = joinpath(@__DIR__, "..", "data", "mppi_trajectories.jld2"),
    hidden_sizes = [32, 32], 
    epochs = 100, 
    learning_rate = 1e-3, 
    num_test_rollouts = 5
)
    # load the mppi trajectories 
    trajectories = load_trajectories(data_file)
    println("trajectories loaded, $(length(trajectories)) MPPI trajectories")

    states, actions = extract_dataset(trajectories)
    N = size(states, 2)
    println("extracted $N state-action pairs ")

    # creation and initialization of the policy 
    policy = NeuralPolicy(env.state_dim, env.action_dim, hidden_sizes)
    params = init_params(policy)
    describe_policy(policy) # just to verify creation

    # train the policy 
    println("training policy")
    final_loss = train_policy!(
        policy, states, actions, params;
        epochs = epochs,
        learning_rate = learning_rate
    )

    @printf("training is complete. final loss: %.4f \n", final_loss)
    println("evaluating policy")

    policy_costs = Float64[]
    for i in 1 : num_test_rollouts
        traj = rollout_policy(env, policy, params; episode_length = 100)
        cost = total_cost(traj)
        push!(policy_costs, cost)
        @printf("rollout %d: cost = %.2f \n", i, cost)
    end 

    # comparison with MPPI 
    mppi_costs = [total_cost(traj) for traj in trajectories]

    println("performance comparison")
    @printf("\nMPPI (expert):\n")
    @printf("  Mean: %.2f ± %.2f\n", mean(mppi_costs), std(mppi_costs))
    @printf("  Min:  %.2f\n", minimum(mppi_costs))

    @printf("\nPolicy (learned):\n")
    @printf("  Mean: %.2f ± %.2f\n", mean(policy_costs), std(policy_costs))
    @printf("  Min:  %.2f\n", minimum(policy_costs))

    gap = mean(policy_costs) - mean(mppi_costs)
    @printf("\n performance gap: %.2f ", gap)

    # save trained policy 
    policy_file = joinpath(@__DIR__, "..", "data", "trained_policy.jld2")
    save_policy(policy_file, policy, params)

    return policy, params
end

function demo_visualize_policy(;
    model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml"),
    policy_file = joinpath(@__DIR__, "..", "data", "trained_policy.jld2")
)
    # load the environment and trained policy
    env = CartpoleEnv(model_path)
    policy, params = load_policy(policy_file)

    # reset with perturbation matching MPPI training distribution
    reset!(env.model, env.data)
    env.data.qpos .+= 0.2 * randn(length(env.data.qpos))
    env.data.qvel .+= 0.2 * randn(length(env.data.qvel))

    # controller function for visualization
    function policy_ctrl!(model, data)
        state = get_physics_state(model, data)
        action = policy(state, params)
        data.ctrl .= clamp.(action, -1.0, 1.0)
        nothing
    end

    # visualize the policy
    init_visualiser()
    Base.invokelatest(visualise!, env.model, env.data, controller = policy_ctrl!)
end


"""Lux-based Neural Policy
"""
struct LuxPolicy
    model::Lux.Chain
    state_dim::Int
    action_dim::Int
end

function LuxPolicy(state_dim::Int, action_dim::Int, hidden_sizes::Vector{Int} = [64, 64])
    # build network using Lux
    model = Lux.Chain(
        Lux.Dense(state_dim => hidden_sizes[1], tanh),
        Lux.Dense(hidden_sizes[1] => hidden_sizes[2], tanh),
        Lux.Dense(hidden_sizes[2] => action_dim, tanh)
    )
    return LuxPolicy(model, state_dim, action_dim)
end

# forward pass returns (output, state)
function (policy::LuxPolicy)(state::AbstractVector, ps, st)
    return policy.model(state, ps, st)
end

# initialize parameters and state
function init_params_lux(policy::LuxPolicy; rng::AbstractRNG = Random.GLOBAL_RNG)
    ps, st = Lux.setup(rng, policy.model)
    return ps, st
end

"""Train Lux policy using Adam optimizer and Zygote for gradients"""
function train_policy_lux!(
    policy::LuxPolicy,
    states::Matrix{Float64},
    actions::Matrix{Float64},
    ps, st;
    epochs::Int = 100,
    learning_rate::Float64 = 1e-3
)
    N = size(states, 2)

    # convert to Float32
    states_f32 = Float32.(states)
    actions_f32 = Float32.(actions)

    # setup Adam optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)

    # training loop
    for epoch in 1:epochs
        # compute loss and gradient using Zygote
        loss_val, grads = Zygote.withgradient(ps) do params
            total_loss = 0.0f0
            for i in 1:N
                pred, _ = policy.model(states_f32[:, i], params, st)
                target = actions_f32[:, i]
                total_loss += sum((pred .- target).^2)
            end
            total_loss / N
        end

        # update parameters with Adam
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        if epoch % 10 == 0
            println("epoch $epoch / $epochs, loss = $(loss_val)")
        end
    end

    # compute final loss
    final_loss = 0.0f0
    for i in 1:N
        pred, _ = policy.model(states_f32[:, i], ps, st)
        target = actions_f32[:, i]
        final_loss += sum((pred .- target).^2)
    end

    return final_loss / N
end

"""Rollout Lux policy in environment"""
function rollout_policy_lux(
    env,
    policy::LuxPolicy,
    ps, st;
    episode_length::Int = 100,
    initial_state = nothing
)
    model = env.model
    data = env.data

    # reset environment
    reset!(model, data)

    if initial_state !== nothing
        nq = length(data.qpos)
        data.qpos .= initial_state[1:nq]
        data.qvel .= initial_state[nq+1:end]
    else
        data.qpos .+= 0.2 * randn(length(data.qpos))
        data.qvel .+= 0.2 * randn(length(data.qvel))
    end

    x0 = get_physics_state(model, data)

    # storage
    states_traj = zeros(env.state_dim, episode_length)
    actions_traj = zeros(env.action_dim, episode_length)
    costs = zeros(episode_length)

    # rollout loop
    for t in 1:episode_length
        state = get_physics_state(model, data)
        states_traj[:, t] = state

        # policy forward pass
        action, _ = policy.model(Float32.(state), ps, st)
        actions_traj[:, t] = action

        # apply action
        data.ctrl .= clamp.(action, -1.0, 1.0)
        step!(model, data)

        # compute cost
        costs[t] = running_cost_cartpole(data)
    end

    costs[end] += terminal_cost_cartpole(data)
    return Trajectory(states_traj, actions_traj, costs, x0)
end

"""Train Lux policy from MPPI trajectories (full pipeline)"""
function train_policy_from_mppi_lux(
    env;
    data_file = joinpath(@__DIR__, "..", "data", "mppi_trajectories.jld2"),
    hidden_sizes = [32, 32],
    epochs = 100,
    learning_rate = 1e-3,
    num_test_rollouts = 5
)
    # load MPPI trajectories
    trajectories = load_trajectories(data_file)
    println("trajectories loaded, $(length(trajectories)) MPPI trajectories")

    states, actions = extract_dataset(trajectories)
    N = size(states, 2)
    println("extracted $N state-action pairs")

    # create and initialize Lux policy
    policy = LuxPolicy(env.state_dim, env.action_dim, hidden_sizes)
    ps, st = init_params_lux(policy)

    # train the policy
    println("\ntraining Lux policy")
    final_loss = train_policy_lux!(
        policy, states, actions, ps, st;
        epochs = epochs,
        learning_rate = learning_rate
    )

    @printf("training complete. final loss: %.4f\n", final_loss)
    println("evaluating policy...")

    # evaluate policy
    policy_costs = Float64[]
    for i in 1:num_test_rollouts
        traj = rollout_policy_lux(env, policy, ps, st; episode_length = 100)
        cost = total_cost(traj)
        push!(policy_costs, cost)
        @printf("rollout %d: cost = %.2f\n", i, cost)
    end

    # comparison with MPPI
    mppi_costs = [total_cost(traj) for traj in trajectories]

    println("\nperformance comparison")
    @printf("\nMPPI (expert):\n")
    @printf("  Mean: %.2f ± %.2f\n", mean(mppi_costs), std(mppi_costs))
    @printf("  Min:  %.2f\n", minimum(mppi_costs))

    @printf("\nLux Policy (learned):\n")
    @printf("  Mean: %.2f ± %.2f\n", mean(policy_costs), std(policy_costs))
    @printf("  Min:  %.2f\n", minimum(policy_costs))

    gap = mean(policy_costs) - mean(mppi_costs)
    @printf("\nperformance gap: %.2f\n", gap)

    return policy, ps, st
end

