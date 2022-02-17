using LinearAlgebra


"""
    kfobservation{T}(obs, obsCov)

The Kalman filter observation and covariance matrices

# Arguments
-`obs`: Observation `Matrix`
-`obsCov`: Observation `Matrix` covariance 

See also [`kfsystem`](@ref), [`kfstate`](@ref)
"""
mutable struct kfobservation{T}
    # Observation matrix
    obs::Matrix{T}
    # Uncertainty on observation
    obsCov::Matrix{T}
end

"""
    kfsystem{T}(transition, control, noise)

The Kalman filter observation and covariance matrices

# Arguments
-`transition`: State transition `Matrix`
-`control`: Input/external control `Matrix `
-`noise`: Estimated noise covariance `Matrix` of the state transition

See also [`kfobservation`](@ref), [`kfstate`](@ref)
"""
mutable struct kfsystem{T}
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
mutable struct kfstate{T}
    # State vector
    state::Vector{T}
    # Covariance matrix
    cov::Matrix{T}

end


"""
    extrapulate_state(state, system, input)

Predict/extrapolate the current state to the next

# Arguments
`state`: [`kfstate`](@ref) instance
`system`: [`kfsystem`](@ref) instance
`input`: External input `Vector`

# Returns
[`kfstate`](@ref)

See also [`update_state`](@ref)
"""
function extrapulate_state(
    state::kfstate,
    system::kfsystem,
    input::Vector)::kfstate
    # Update the state vector
    updated_state_vec::Vector = system.transition * state.state + system.control * input
    updated_cov::Matrix = system.transition * state.cov * transpose(system.transition)
                                + system.noise * input

    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_cov)
    return updated_state
end

"""
    update_state(predicted_state, current_filter, measurement)

Update the current state estimate with a measurement

# Arguments
`predicted_state`: [`kfstate`](@ref) instance after running [`extrapulate_state`](@ref)
`observation`: [`kfobservation`](@ref) instance dictating the observation of the system
`measurement`: The measurement `Vector`

# Returns
[`kfstate`](@ref)

See also [`extrapulate_state`](@ref)
"""
function update_state(
    predicted_state::kfstate,
    observation::kfobservation,
    measurement::Vector)::kfstate
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
                         * (measurement - observation.obs * predicted_state.state))
    updated_cov = (transformation * predicted_state.cov * transpose(transformation)
                   + gain * observation.obsCov * transpose(gain))
    updated_state = kfstate{eltype(updated_state_vec)}(updated_state_vec, updated_cov)

    return updated_state
end
