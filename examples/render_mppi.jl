using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch
using Printf

"""
Render cartpole trajectories in MuJoCo viewer.

Modes:
1. Live MPPI: Run MPPI control in real-time
2. Replay: Load and replay saved trajectories

Usage:
  julia render_mppi.jl              # Run live MPPI
  julia render_mppi.jl replay       # Replay saved trajectories
  julia render_mppi.jl replay 2     # Replay trajectory #2
"""

function render_live_mppi(env, episode_length=500)

    # setup MPPI planner
    T = 20
    planner = MPPIPlanner(env.action_dim, T)
    println("✓ MPPI planner created (horizon = $T)")

    # reset with initial noise
    reset!(env.model, env.data)
    env.data.qpos .+= 0.3 * randn(length(env.data.qpos))
    env.data.qvel .+= 0.1 * randn(length(env.data.qvel))

    println("\n running MPPI control")
    println("Close viewer window to stop.")

    # define MPPI controller for visualizer
    function mppi_ctrl!(model, data)
        mppi_controller!(env, planner)
        nothing
    end

    # initialize and run visualizer
    init_visualiser()
    Base.invokelatest(visualise!, env.model, env.data, controller=mppi_ctrl!)

    println("\n✓ Visualization complete")
end

function render_saved_trajectory(env, traj_idx=1)
    println("\n" * "="^60)
    println("Replaying Saved Trajectory")
    println("="^60)

    # load trajectories
    data_file = joinpath(@__DIR__, "..", "data", "mppi_trajectories.jld2")

    if !isfile(data_file)
        println("no saved trajectories found at: $data_file")
        println("run vis_mppi.jl first to generate trajectories.")
        return
    end

    trajectories = load_trajectories(data_file)

    if traj_idx > length(trajectories)
        println("trajectory $traj_idx not found (only $(length(trajectories)) available)")
        return
    end

    traj = trajectories[traj_idx]
    traj_length = trajectory_length(traj)

    println(" Loaded trajectory $traj_idx / $(length(trajectories))")
    @printf("  Length: %d steps\n", traj_length)
    @printf("  Total cost: %.2f\n", total_cost(traj))

    # reset to initial state
    reset!(env.model, env.data)
    env.data.qpos .= traj.x0[1:length(env.data.qpos)]
    env.data.qvel .= traj.x0[length(env.data.qpos)+1:end]

    println("\nReplaying trajectory...")
    println("Close viewer window to stop.")

    # visualize trajectory playback
    init_visualiser()
    Base.invokelatest(visualise!, env.model, env.data; trajectories=[traj.states])

    println("\n✓ Replay complete")
end

function main()
    println("MuJoCo Cartpole Visualization")


    # setup environment
    model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml")
    env = CartpoleEnv(model_path)

    # parse command line arguments
    mode = length(ARGS) >= 1 ? ARGS[1] : "live"

    if mode == "replay"
        traj_idx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
        render_saved_trajectory(env, traj_idx)
    else
        render_live_mppi(env)
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
