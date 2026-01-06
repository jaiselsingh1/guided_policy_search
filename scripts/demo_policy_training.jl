using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch

# run GPS iteration loop
model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml")
env = CartpoleEnv(model_path)

policy, ps, st, history = guided_policy_search(
    env;
    num_iterations=3,
    trajectories_per_iter=20,
    policy_epochs=50,
    learning_rate=1e-3,
    num_eval_rollouts=5
)

println("\nGPS complete!")
