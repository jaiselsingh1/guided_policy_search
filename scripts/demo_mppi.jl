using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch

# generate and save MPPI trajectories
# demo_generate_and_save(num_trajectories=5, trajectory_length=100)

# visualize saved trajectories
# demo_visualize_saved()

# live MPPI control
demo_live_mppi()
