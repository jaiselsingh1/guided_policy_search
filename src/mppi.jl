using MuJoCo
using LinearAlgebra
using Base.Threads
using Random

# Helper function to extract full physics state [qpos; qvel]
function get_physics_state(model::Model, data::Data)
    # explicitly create a Vector to avoid MuJoCo array type issues
    nq = length(data.qpos)
    nv = length(data.qvel)
    state = Vector{Float64}(undef, nq + nv)
    state[1:nq] .= data.qpos
    state[nq+1:end] .= data.qvel
    return state
end 

struct CartpoleEnv 
    model::Model 
    data::Data 
    action_dim::Int 
    state_dim::Int 
end 

function CartpoleEnv(model_path::String)
    model = load_model(model_path)
    data = init_data(model)
    action_dim = model.nu
    state_dim = length(get_physics_state(model, data))

    return CartpoleEnv(model, data, action_dim, state_dim)
end 

struct HopperEnv
    model::Model 
    data::Data 
    action_dim::Int 
    state_dim::Int
end 

function HopperEnv(model_path::String)
    model = load_model(model_path)
    data = init_data(model)
    action_dim = model.nu
    state_dim = length(get_physics_state(model, data))

    return HopperEnv(model, data, action_dim, state_dim)
end


# cost functions 
function running_cost_cartpole(data::Data)
    ctrl = data.ctrl
    x = data.qpos[1]
    θ = data.qpos[2]
    x_dot = data.qvel[1]
    θ_dot = data.qvel[2]

    pos_cost = 1.0 * x^2
    theta_cost = 100.0 * (cos(θ) - 1)^2 
    vel_cost = 0.1 * x_dot^2
    thetadot_cost = 1.0 * θ_dot^2
    ctrl_cost = ctrl[1]^2
    return pos_cost + theta_cost + vel_cost + thetadot_cost + ctrl_cost
end 

function terminal_cost_cartpole(data::Data)
    return 10.0 * running_cost_cartpole(data)
end

# MPPI Planner 
"""This is the main MPPI data structure
Fields 
- U (control sequence) --> (action_dim x T)
- T (planning horizon)
"""

# the mutable allows the core values that are within the U and T fields to be updated
mutable struct MPPIPlanner
    U::Matrix{Float64} # this is (action_dim x T)
    T::Int
end 

function MPPIPlanner(action_dim::Int, T::Int, init_std::Float64 = 0.2)
    U = init_std .* randn(action_dim, T)
    return MPPIPlanner(U, T)
end 

# mppi_step!() performs one core MPPI planning step 
"""
# Arguments
  - `env`: Environment to plan in
  - `planner`: MPPI planner with current control sequence
  - `K`: Number of sample rollouts (default 100)
  - `λ`: Temperature parameter (default 1.0)
  - `Σ`: Noise covariance (default 1.0)

  # Algorithm
  1. Sample K noisy control sequences around current U
  2. Rollout each sequence and compute cost
  3. Weight samples by exp(-cost/λ)
  4. Update U as weighted average
"""

function mppi_step!(
    env::CartpoleEnv,
    planner::MPPIPlanner,
    K::Int = 500, # the number of samples
    λ::Float64 = 1.0, # the temperature parameter
    Σ::Float64 = 1.0 # the noise std
)

    action_dim = env.action_dim
    U = planner.U 
    T = planner.T
    model = env.model
    data = env.data

    # sample noise for each K
    ϵ = [randn(action_dim, T) for _ in 1:K]
    S = zeros(K) # costs are initialized as 0s 

    # each thread gets its own mj_data
    local_datas = [init_data(model) for _ in 1:nthreads()]

    # parallel rollouts
    @threads for k in 1:K
        # any thread can be running multiple samples
        local_d = local_datas[threadid()]
        
        # reset to the current state 
        # do i need a copy here?
        local_d.qpos .= data.qpos 
        local_d.qvel .= data.qvel 
        
        # rollout with noisy controls 
        for t in 1:T
            # apply control with noise but then clamp to [-1, 1]
            noisy_control = clamp.(U[:, t] + Σ * ϵ[k][:, t], -1.0, 1.0)
            local_d.ctrl .= noisy_control 
            step!(model, local_d)

            # accumulate cost 
            step_cost = running_cost_cartpole(local_d)

            # mppi control cost term 
            S[k] += step_cost + (λ / Σ) * dot(U[:, t], ϵ[k][:, t])

        end

        # add the terminal cost 
        S[k] += terminal_cost_cartpole(local_d)

    end 

    # compute the importance weights 
    β = minimum(S) # this is the baseline for numerical stability 
    # the temperature tells you how much the cost differences are going to impact the weights
    weights = exp.((-1.0 / λ) .* (S .- β))
    # think of this as a softmin over the costs that's controlled by λ
    weights ./= sum(weights)

    for t in 1:T
        planner.U[:, t] .+= sum(weights[k] * ϵ[k][:, t] for k in 1:K)
    end 

end 

"""
mppi_controller!(env::CartpoleEnv, planner::MPPIPlanner)
Execute one step of MPPI control in the environment.

  # Steps
  1. Plan with MPPI (optimize U)
  2. Apply first control U[:, 1]
  3. Shift control sequence (for next timestep)
  """

function mppi_controller!(env::CartpoleEnv, planner::MPPIPlanner)
    mppi_step!(env, planner)
    env.data.ctrl .= planner.U[:, 1]

    # shift the planner 
    planner.U[:, 1:end-1] .= planner.U[:, 2:end]
    planner.U[:, end] .= 0.0

    return nothing
end 

# trajectory generation
function generate_trajectories(
    env::CartpoleEnv,
    planner::MPPIPlanner;
    num_trajectories::Int = 10,
    trajectory_length::Int = 100,
    initial_state_noise::Float64 = 0.2,
)
    model = env.model 
    data = env.data 
    trajectories = Trajectory[]

    for traj_idx in 1:num_trajectories
        # reset the model with random initial condition 
        reset!(model, data)
        data.qpos .+= initial_state_noise * randn(length(data.qpos))
        data.qvel .+= initial_state_noise * randn(length(data.qvel))

        x0 = get_physics_state(model, data)
        # reset planner with new random initialization
        planner.U .= 0.2 * randn(env.action_dim, planner.T)
        
        # storage for this trajectory
        states = zeros(env.state_dim, trajectory_length)
        actions = zeros(env.action_dim, trajectory_length)
        costs = zeros(trajectory_length)

        # rollout with MPPI control
        for t in 1:trajectory_length
            states[:, t] = get_physics_state(model, data)

            # MPPI plans and applies control
            mppi_controller!(env, planner)

            actions[:, t] = data.ctrl
            costs[t] = running_cost_cartpole(data)

            # Step simulation
            step!(model, data)
        end

        # add terminal cost
        costs[end] += terminal_cost_cartpole(data)

        # create trajectory object
        traj = Trajectory(states, actions, costs, x0)
        push!(trajectories, traj)

        println("generated trajectory $traj_idx / $num_trajectories, cost = $(total_cost(traj))")
    end
    return trajectories

end

# demo/testing functions
function demo_generate_and_save(;
    model_path=joinpath(@__DIR__, "..", "models", "cartpole.xml"),
    num_trajectories=20,
    trajectory_length=100
)
    env = CartpoleEnv(model_path)
    planner = MPPIPlanner(env.action_dim, 40)

    trajectories = generate_trajectories(
        env, planner;
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        initial_state_noise=0.2
    )

    data_dir = joinpath(@__DIR__, "..", "data")
    !isdir(data_dir) && mkdir(data_dir)
    save_trajectories(joinpath(data_dir, "mppi_trajectories.jld2"), trajectories)

    costs = [total_cost(traj) for traj in trajectories]
    @printf("\nMean: %.2f, Min: %.2f, Max: %.2f\n",
            sum(costs)/length(costs), minimum(costs), maximum(costs))

    return trajectories
end

function demo_visualize_saved(;
    model_path=joinpath(@__DIR__, "..", "models", "cartpole.xml"),
    data_file=joinpath(@__DIR__, "..", "data", "mppi_trajectories.jld2")
)
    env = CartpoleEnv(model_path)
    trajectories = load_trajectories(data_file)

    init_visualiser()
    traj_states = [traj.states for traj in trajectories]
    Base.invokelatest(visualise!, env.model, env.data; trajectories=traj_states)
end

function demo_live_mppi(;
    model_path=joinpath(@__DIR__, "..", "models", "cartpole.xml")
)
    env = CartpoleEnv(model_path)
    planner = MPPIPlanner(env.action_dim, 40)

    reset!(env.model, env.data)
    env.data.qpos .+= 0.2 * randn(length(env.data.qpos))
    env.data.qvel .+= 0.2 * randn(length(env.data.qvel))

    function mppi_ctrl!(model, data)
        mppi_controller!(env, planner)
        nothing
    end

    init_visualiser()
    Base.invokelatest(visualise!, env.model, env.data, controller=mppi_ctrl!)
end
