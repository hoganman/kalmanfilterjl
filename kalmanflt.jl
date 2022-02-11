using LinearAlgebra


mutable struct kfobservation{T}
    """Kalman filter observation and covariance matrices"""

    # Observation matrix
    obs::Matrix{T}
    # Uncertainty on observation
    obsCov::Matrix{T}

end

mutable struct kfsystem{T}
    """Kalman filter state transition, control input, and noise covariance matrices"""

    # State transition matrix
    transition::Matrix{T}
    # Control matrix
    control::Matrix{T}
    # Input noise
    noise::Matrix{T}

end

mutable struct kfstate{T}
    """Kalman filter state vector and covariance matrix"""

    # State vector
    state::Vector{T}
    # Covariance matrix
    cov::Matrix{T}

end


function extrapulate_state(
    state::kfstate,
    system::kfsystem,
    input::Vector)::kfstate
    """Extrapolate the current state to the next

    :param state: Current state
    :param system: System matrices
    :param input: Input vector
    :returns: kfstate
    """

    # Update the state vector
    updated_state_vec::Vector = system.transition * state.state + system.control * input
    updated_cov::Matrix = system.transition * state.cov * transpose(system.transition)
                                + system.noise * input

    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_cov)
    return updated_state
end

function update_state(
    predicted_state::kfstate,
    current_filter::kfobservation,
    measurement::Vector)::kfstate
    """Update the predicted state with the current measurement

    :param predicted_state: Predicted state
    :param current_filter: Observation and associated covariance matrix
    :param measurement: Measurement vector
    :returns: kfstate
    """

    # Intermediate calculation (1)
    obs_transpose = transpose(current_filter.obs)

    # Compute the Kalman gain
    gain = (predicted_state.cov * obs_transpose
            * (current_filter.obs * predicted_state.cov * obs_transpose
               + current_filter.obsCov)^(-1))

    # Intermediate calculation (2)
    transformation = (I(size(predicted_state.cov)[1]) - gain * current_filter.obs)

    # Update the state estimate and covariance
    updated_state_vec = (predicted_state.state + gain
                         * (measurement - current_filter.obs * predicted_state.state))
    updated_cov = (transformation * predicted_state.cov * transpose(transformation)
                   + gain * current_filter.obsCov * transpose(gain))
    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_cov)

    return updated_state
end
