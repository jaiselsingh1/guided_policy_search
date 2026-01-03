using MuJoCo 
using LinearAlgebra 
using Base.Threads
using Random 

struct CartpoleEnv 
    model::Model 
    data::Data 
    action_dim::Int 
    state_dim::Int 
end 

function CartpoleEnv(model_path::String)
    model = load_model(model_path)
    data = init_data(model)
    action_dim = model.ν
    state_dim = length(get_phsics_state(model, data))

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
    action_dim = model.ν
    state_dim = length(get_phsics_state(model, data))

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
    theta_cost = 20.0 * (cos(θ) - 1)^2 
    vel_cost = 0.1 * x_dot^2
    thetadot_cost = 0.1 * θ_dot^2
    ctrl_cost = ctrl[1]^2

    return pos_cost + theta_cost + vel_cost + thetadot_cost + ctrl_cost
end 

function terminal_cost_cartpole(data::Data)
    return 10.0 * running_cost_cartpole(data)
end




