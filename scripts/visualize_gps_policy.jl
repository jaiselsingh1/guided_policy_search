using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GuidedPolicySearch

# train GPS and visualize the final policy
model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml")
env = CartpoleEnv(model_path)

println("training GPS...")
policy, ps, st, history = guided_policy_search(
    env;
    num_iterations=3,
    trajectories_per_iter=20,
    policy_epochs=50,
    learning_rate=1e-3,
    num_eval_rollouts=5,
    kl_weight=0.1
)

println("\n visualizing trained policy")

# reset with perturbation
reset!(env.model, env.data)
env.data.qpos .+= 0.2 * randn(length(env.data.qpos))
env.data.qvel .+= 0.2 * randn(length(env.data.qvel))

# controller function for visualization
function policy_ctrl!(model, data)
    state = get_physics_state(model, data)
    output, _ = policy.model(Float32.(state), ps, st)

    # extract mean from stochastic policy
    action = output[1:policy.action_dim]
    data.ctrl .= clamp.(action, -1.0, 1.0)
    nothing
end

# visualize the policy
init_visualiser()
Base.invokelatest(visualise!, env.model, env.data, controller = policy_ctrl!)
