# GPS iteration functions will be implemented here

function guided_policy_search(
    env;
    num_iterations::Int = 10,
    trajectories_per_iter::Int = 20,
    trajectory_length::Int = 100,
    initial_state_noise::Float64 = 0.2,
    hidden_sizes::Vector{Int} = [32, 32],
    policy_epochs::Int = 50,
    learning_rate::Float64 = 1e-3,
    num_eval_rollouts::Int = 5,
    kl_weight::Float64 = 0.1
)
    # initialize the stochastic policy
    policy = LuxPolicy(env.state_dim, env.action_dim, hidden_sizes; stochastic=true)
    ps, st = init_params_lux(policy)

    # tracking 
    history = Dict(
        "mppi_costs" => Vector{Float64}[], 
        "policy_costs" => Vector{Float64}[],
        "training_losses" => Float64[]
    )

    planner = MPPIPlanner(env.action_dim, 40)

    for iter in 1 : num_iterations
        println("GPS iteration $iter / $num_iterations")

        # generate trajectories with MPPI (warm-start with policy after first iteration)
        if iter == 1
            # first iteration: random initialization
            trajectories = generate_trajectories(
                env, planner;
                num_trajectories = trajectories_per_iter,
                trajectory_length = trajectory_length,
                initial_state_noise = initial_state_noise
            )
        else
            # subsequent iterations: warm-start with policy
            trajectories = generate_trajectories(
                env, planner;
                num_trajectories = trajectories_per_iter,
                trajectory_length = trajectory_length,
                initial_state_noise = initial_state_noise,
                policy = policy,
                ps = ps,
                st = st
            )
        end

        mppi_costs = [total_cost(traj) for traj in trajectories]
        push!(history["mppi_costs"], mppi_costs)

        @printf("MPPI: mean = %.2f, std = %.2f, min = %.2f\n", mean(mppi_costs), std(mppi_costs), minimum(mppi_costs))

        # extract training data 
        states, actions = extract_dataset(trajectories)
        @printf("Extracted %d state-action pairs\n", size(states, 2))

        # train the policy with KL constraint
        ps, st, final_loss = train_policy_lux_kl!(
            policy, states, actions, ps, st;
            epochs = policy_epochs,
            learning_rate = learning_rate,
            kl_weight = kl_weight
        )

        push!(history["training_losses"], final_loss)
        @printf("Training loss: %.4f\n", final_loss)

        # eval policy
        policy_costs = Float64[]
        for i in 1 : num_eval_rollouts
            traj = rollout_policy_lux(env, policy, ps, st; episode_length = trajectory_length)
            push!(policy_costs, total_cost(traj))
        end 
        push!(history["policy_costs"], policy_costs)

        gap = mean(policy_costs) - mean(mppi_costs)
        @printf("Performance gap: %.2f\n", gap)
    end 
    println("gps complete")
    return policy, ps, st, history

end 