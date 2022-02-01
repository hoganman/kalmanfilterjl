using LinearAlgebra


mutable struct klobservation{T}

    # Observation matrix
    obs::Matrix{T}
    # Uncertainty on observation
    obsCov::Matrix{T}

end

mutable struct klsystem{T}

    # State transition matrix
    transition::Matrix{T}
    # Control matrix
    control::Matrix{T}
    # Input noise
    noise::Matrix{T}

end

mutable struct klstate{T}
    
    # State vector
    state::Vector{T}
    # Covariance matrix
    cov::Matrix{T}
    
end


function extrapulate_state(
    state::klstate,
    system::klsystem,
    input::Vector)::klstate
    
    # Update the state vector
    updated_state::Vector = system.transition * state.state + system.control * input
    updated_cov::Matrix = system.transition * state.cov * transpose(system.transition) 
                                + system.noise * input
end

function update_state(
    current_state::klstate,
    current_filter::klobservation,
    measurement::Vector)::klstate

    # Intermediate calculation (1)
    obs_transpose = transpose(current_filter.obs)

    # Compute the Kalman gain
    gain = (current_state.cov * obs_transpose
            * (current_filter.obs * current_state.cov * obs_transpose
               + current_filter.obsCov)^(-1))

    # Intermediate calculations (2)
    transformation = (ones(eltype(current_state.cov), size(current_state.cov))
                      - gain * current_filter.obs)

    # Update the state estimate and covariance
    updated_state_vec = (current_state.state + gain 
                         * (measurement - current_filter.obs * current_state.state))
    updated_cov = (transformation * current_state.cov * transpose(transformation)
                   + gain * current_filter.obsCov * transpose(gain))
    updated_state = klstate(updated_state_vec, updated_cov)
    
    return updated_state
end