using Statistics
using JLD2

"""
Trajectory Data Structure for storing the rollout information
"""
struct Trajectory
    states::Matrix{Float64} # state dim x T
    actions::Matrix{Float64} # action dim x T
    costs::Vector{Float64} # (T, )
    x0::Vector{Float64} # (state_dim, )
end 

# helper functions 
function total_cost(traj::Trajectory)
    return sum(traj.costs)
end 

function trajectory_length(traj::Trajectory)
    return length(traj.costs)
end 

# extract all state, action pairs from trajectories for SL
function extract_dataset(trajectories::Vector{Trajectory})
    all_states = hcat([traj.states for traj in trajectories]...)
    all_actions = hcat([traj.actions for traj in trajectories]...)
    return all_states, all_actions 
end 

# save / load trajectories 
function save_trajectories(filename::String, trajectories::Vector{Trajectory})
    initial_states = [traj.x0 for traj in trajectories]
    JLD2.@save filename trajectories initial_states 
    println("saved  $(length(trajectories)) trajectories to $filename")
end 

function load_trajectories(filename::String)
    JLD2.@load filename trajectories initial_states
    println("loaded $(length(trajectories)) trajectories from $filename")
    return trajectories
end 