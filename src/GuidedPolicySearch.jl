module GuidedPolicySearch

# core dependencies
using MuJoCo
using LinearAlgebra
using Statistics
using Random
using Distributions
using SimpleChains
using Lux
using Optimisers
using Zygote
using JLD2
using Printf

# include all source files
include("trajectory.jl")
include("mppi.jl")
include("policy.jl")
include("gps.jl")

# export trajectory types and functions
export Trajectory
export total_cost, trajectory_length, extract_dataset
export save_trajectories, load_trajectories

# export MPPI types and functions
export CartpoleEnv, HopperEnv
export MPPIPlanner
export mppi_step!, mppi_controller!, generate_trajectories
export running_cost_cartpole, terminal_cost_cartpole
export get_physics_state
export demo_generate_and_save, demo_visualize_saved, demo_live_mppi

# re-export commonly used MuJoCo functions
export reset!, step!, load_model, init_data, visualise!, init_visualiser

# export policy types and functions
export NeuralPolicy, LuxPolicy
export init_params, train_policy!
export init_params_lux, train_policy_lux!
export save_policy, load_policy
export describe_policy
export rollout_policy, rollout_policy_lux
export train_policy_from_mppi, train_policy_from_mppi_lux
export demo_visualize_policy

# GPS functions will be exported once implemented
# export guided_policy_search, gps_iteration

end
