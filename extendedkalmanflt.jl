using SymPy: Sym, diff
include("kalmanflt.jl")


struct StateVariable

    pred_func::Any

    prev_state_sym::Sym

    pred_func_syms::Array{Sym}

end

struct ObservationVariable
    
    obs_func::Any
    
    obs_func_syms::Array{Sym}
    
end


function eval_jacobian_sym_mat(
    mat::Matrix{Sym}, 
    par_values_dict::Dict
)::Matrix{Float64}
   jacobian = zeros(Float64, size(mat))
    for row in 1:size(jacobian)[1]
        for col in 1:size(jacobian)[2]
            jacobian[row, col] = float(
                mat[row, col].subs(par_values_dict).evalf()
            )
        end
    end 
    return jacobian
end


function eval(
    mat::Matrix{Sym}, 
    par_values_dict::Dict
)
    return eval_jacobian_sym_mat(mat, par_values_dict)
end


function observation_matrix_sym(
    observation_variables::Vector{ObservationVariable},
    state_variables::Vector{StateVariable}
)::Matrix{Sym}

    unevaluated_jacobian = Matrix{Sym}(undef, (length(observation_variables), length(state_variables)))
    for row in 1:length(observation_variables)
        obs_variable = observation_variables[row].obs_func
        obs_fcn_variables = observation_variables[row].obs_func_syms
        for col in 1:length(state_variables)
            state_n_pred_variable = state_variables[col].prev_state_sym
            deriv = diff(obs_variable(obs_fcn_variables...), state_n_pred_variable, evaluate=false)
            unevaluated_jacobian[row, col] = deriv
        end
    end
    return unevaluated_jacobian
end


function transition_matrix_sym(
    state_variables::Vector{StateVariable}
)::Matrix{Sym}

    unevaluated_jacobian = Matrix{Sym}(
        undef, (length(state_variables), length(state_variables))
    )
    for row in 1:length(state_variables)
        state_n_variable = state_variables[row].pred_func
        state_fcn_variables = state_variables[row].pred_func_syms
        for col in 1:length(state_variables)
            state_nm1_variable = state_variables[col].prev_state_sym
            deriv = diff(
                state_n_variable(state_fcn_variables...), 
                state_nm1_variable, 
                evaluate=false
            )
            unevaluated_jacobian[row, col] = deriv
        end
    end
    return unevaluated_jacobian
end



function predict_state(
    state_vars_vec::Vector{StateVariable},
    prev_state::KalmanState,
    model_vars::Dict,
    noise_mat::Matrix
)::KalmanState

    index::Int
    eval_dict = model_vars
    pred_state_vec = zeros(eltype(prev_state), size(state_vars_vec))
    for index in 1:length(state_vars_vec)
        key::Sym = state_vars_vec[index].prev_state_sym
        value = prev_state.state[index]
        eval_dict[key] = value
    end

    trans_mat::Matrix = eval(transition_matrix_sym(state_vars_vec), eval_dict)
    for index in 1:length(state_vars_vec)
        state_var::StateVariable = state_vars_vec[index]
        pred_func_sym::Sym = state_var.pred_func(state_var.pred_func_syms...)
        pred_state_vec[index] = pred_func_sym.subs(eval_dict).evalf()
    end
    pred_state_cov = (trans_mat * prev_state.cov * transpose(trans_mat)
                      + noise_mat)
    return KalmanState{eltype(pred_state_cov)}(pred_state_vec, pred_state_cov)
end


function correct_state(
    state_vars_vec::Vector{StateVariable},
    obs_vars_vec::Vector{ObservationVariable},
    model_vars::Dict,
    pred_state::KalmanState,
    measurement::KalmanObservation
)::KalmanState

    index::Int
    eval_dict = model_vars
    for index in 1:length(state_vars_vec)
        key::Sym = state_vars_vec[index].prev_state_sym
        value = pred_state.state[index]
        eval_dict[key] = value
    end

    obs_matrix::Matrix = measurement.obs
    if all(els -> els == 0, obs_matrix)
        obs_matrix = eval(
            observation_matrix_sym(
                obs_vars_vec,
                state_vars_vec
            ), eval_dict
        )
    end
    meas_residual = zeros(eltype(measurement.meas), size(measurement.meas))
    for index in 1:length(obs_vars_vec)
        obs_var = obs_vars_vec[index]
        obs_pred = obs_var.obs_func(obs_var.obs_fcn_variables...).subs(eval_dict).evalf()
        meas_residual[index] = measurement.meas[index] - obs_pred
    end
    residual_cov = (obs_matrix * pred_state.cov * transpose(obs_matrix)
                    + measurement.obsCov)
    gain = pred_state.cov * transpose(obs_matrix) * residual_cov^(-1)

    corr_state_vec = pred_state.state + gain * meas_residual
    corr_cov = (I(length(corr_state_vec)) - gain * obs_matrix) * pred_state.cov
    return KalmanState{eltype(corr_cov)}(corr_state_vec, corr_cov)
end;
