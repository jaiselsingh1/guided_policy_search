using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch

# generate MPPI trajectories (if not already done)
# demo_generate_and_save(num_trajectories=10, trajectory_length=100)

# train policy on MPPI data
model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml")
env = CartpoleEnv(model_path)

policy, params = train_policy_from_mppi(
    env;
    epochs=100,
    learning_rate=1e-3,
    num_test_rollouts=5
)

println("\n policy training complete!")


println("\n visualizing trained policy")
demo_visualize_policy()
