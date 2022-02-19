using LinearAlgebra


"""
    kfobservation{T}(obs, obsCov)

The Kalman filter observation and covariance matrices

# Arguments
-`obs`: Observation `Matrix`
-`obsCov`: Observation `Matrix` covariance 

See also [`kfsystem`](@ref), [`kfstate`](@ref)
"""
mutable struct kfobservation{T <: Real}

    # Measurement vector
    meas::Vector{T}
    # Observation matrix
    obs::Matrix{T}
    # Uncertainty on observation
    obsCov::Matrix{T}

end

kfobservation{T}(obs::Matrix{T}, obsCov::Matrix{T}) where {T<:Real} = kfobservation{T}(
    zeros(T, size(obs)[1]), 
    obs,
    obsCov
);

"""
kfupdate{T}(transition, control, noise)

The Kalman filter transition, control, and noise covariance matrices, collectively the update matrices

# Arguments
 -`transition`: State transition `Matrix`
 -`control`: Input/external control `Matrix `
 -`noise`: Estimated noise covariance `Matrix` of the state transition

See also [`kfobservation`](@ref), [`kfstate`](@ref)
"""
struct kfupdate{T <: Real}
    # State transition matrix
    transition::Matrix{T}
    # Control matrix
    control::Matrix{T}
    # Input noise
    noise::Matrix{T}
end

"""
    kfstate{T}(state, cov)

The Kalman filter state vector and associated covariance matrix

# Arguments
 -`state`: State `Vector`
 -`cov`: Covariance `Matrix` of the state

See also [`kfobservation`](@ref), [`kfsystem`](@ref)
"""
struct kfstate{T <: Real}
    # State vector
    state::Vector{T}
    # Covariance matrix
    cov::Matrix{T}

end


"""
predict_state(state, system, input)

Predict the next state vector and covariance

# Arguments
 -`state`: [`kfstate`](@ref) instance
 -`update_matrices`: [`kfupdate`](@ref) instance
 -`input`: External input `Vector`

# Returns
[`kfstate`](@ref)

See also [`update_state`](@ref)
"""
function predict_state(
    state::kfstate,
    update_matrices::kfupdate,
    external_input::Vector
)::kfstate
    # Update the state vector
    updated_state_vec::Vector = (update_matrices.transition * state.state 
                                 + update_matrices.control * external_input)
    updated_state_cov::Matrix = ((update_matrices.transition * state.cov 
                           * transpose(update_matrices.transition)) + update_matrices.noise)

    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_state_cov)
    return updated_state
end

"""
    correct_state(predicted_state, current_filter, measurement)

Update the current state estimate with a measurement

# Arguments
 -`predicted_state`: [`kfstate`](@ref) instance after running [`extrapulate_state`](@ref)
 -`observation`: [`kfobservation`](@ref) instance dictating the observation of the system
 -`measurement`: The measurement `Vector`

# Returns
[`kfstate`](@ref)

See also [`extrapulate_state`](@ref)
"""
function correct_state(
    predicted_state::kfstate,
    observation::kfobservation
)::kfstate
    # Intermediate calculation (1)
    obs_transpose = transpose(observation.obs)

    # Compute the Kalman gain
    gain = (predicted_state.cov * obs_transpose
            * (observation.obs * predicted_state.cov * obs_transpose
               + observation.obsCov)^(-1))

    # Intermediate calculation (2)
    transformation = (I(size(predicted_state.cov)[1]) - gain * observation.obs)

    # Update the state estimate and covariance
    updated_state_vec = (predicted_state.state + gain
                         * (observation.meas - observation.obs * predicted_state.state))
    updated_cov = (transformation * predicted_state.cov * transpose(transformation)
                   + gain * observation.obsCov * transpose(gain))
    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_cov)

    return updated_state
end
