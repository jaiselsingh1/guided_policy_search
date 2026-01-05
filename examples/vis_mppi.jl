using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch
using Printf

function main()
    # a @ sign is a "macro" which is something that runs at parse time and rewrites itself when executing
    # parse time is when julia creates an AST -> abstract syntax tree
    # set up the environment 
    model_path = joinpath(@__DIR__, "..", "models","cartpole.xml")
    env = CartpoleEnv(model_path)
    println("loaded cartpole env")
    
    # set up the MPPI planner 
    T = 20 
    planner = MPPIPlanner(env.action_dim, T)
    println("created MPPI Planner, where the horizon is $T")

    # generate the trajectories 
    num_trajs = 5 
    traj_length = 100
    initial_noise = 0.7 

    trajectories = generate_trajectories(
        env, 
        planner; 
        num_trajectories = num_trajs, 
        trajectory_length = traj_length, 
        initial_state_noise = initial_noise
    )

    # analyse the trajectories 
    costs = [total_cost(traj) for traj in trajectories]
    for (i, cost) in enumerate(costs)
        @printf("  Trajectory %d: %.2f\n", i, cost)
    end 

    println("\n Summary")
    @printf("  mean cost: %.2f\n", sum(costs) / length(costs))
    @printf("  min cost:  %.2f\n", minimum(costs))
    @printf("  max cost:  %.2f\n", maximum(costs))

    # analyse the state distributions 
    all_states = hcat([traj.states for traj in trajectories]...)
    println("\n state statistics")
    state_names = ["x (cart pos)", "θ (pole angle)", "ẋ (cart vel)", "θ̇ (pole vel)"]
    for i in 1:env.state_dim
        state_values = all_states[i, :]
        @printf(" %s: mean=%.3f, std=%.3f, range = [%.3f, %.3f]\n", 
        state_names[i], sum(state_values) / length(state_values), 
        sqrt(sum((state_values .- sum(state_values) / length(state_values)).^2) /  length(state_values)),
        minimum(state_values), 
        maximum(state_values))
    end

    data_dir = joinpath(@__DIR__, "..", "data")
    if !isdir(data_dir)
        mkdir(data_dir)
    end 
    output_file = joinpath(data_dir, "mppi_trajectories.jld2")
    save_trajectories(output_file, trajectories)
    println("saved trajectories to $output_file")
 
end 

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end