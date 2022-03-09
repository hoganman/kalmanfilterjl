using LinearAlgebra


"""
    KalmanObservation{T}(obs, obsCov)

The Kalman filter observation and covariance matrices

# Arguments
-`obs`: Observation `Matrix`
-`obsCov`: Observation `Matrix` covariance 

See also [`kfsystem`](@ref), [`kfstate`](@ref)
"""
mutable struct KalmanObservation{T <: Real}

    # Measurement vector
    meas::Vector{T}
    # Observation matrix
    obs::Matrix{T}
    # Uncertainty on observation
    obsCov::Matrix{T}

end

KalmanObservation{T}(
        obs::Matrix{T}, 
        obsCov::Matrix{T}
) where {T<:Real} = KalmanObservation{T}(
    zeros(T, size(obs)[1]), 
    obs,
    obsCov
);

KalmanObservation{T}(
        meas::Vector{T}, 
        obsCov::Matrix{T}
) where {T<:Real} = KalmanObservation{T}(
    meas, 
    zeros(eltype(obsCov), size(obsCov)),
    obsCov
);

"""
    KalmanUpdate{T}(transition, control, noise)

The Kalman filter transition, control, and noise covariance matrices, collectively the update matrices

# Arguments
 -`transition`: State transition `Matrix`
 -`control`: Input/external control `Matrix `
 -`noise`: Estimated noise covariance `Matrix` of the state transition

See also [`kfobservation`](@ref), [`kfstate`](@ref)
"""
struct KalmanUpdate{T <: Real}
    # State transition matrix
    transition::Matrix{T}
    # Control matrix
    control::Matrix{T}
    # Input noise
    noise::Matrix{T}
end

"""
    KalmanState{T}(state, cov)

The Kalman filter state vector and associated covariance matrix

# Arguments
 -`state`: State `Vector`
 -`cov`: Covariance `Matrix` of the state

See also [`kfobservation`](@ref), [`kfsystem`](@ref)
"""
struct KalmanState{T <: Real}
    # State vector
    state::Vector{T}
    # Covariance matrix
    cov::Matrix{T}

end

KalmanState{T}(
        ks::KalmanState
) where {T<:Real} = begin
    KalmanState(ks.state, ks.cov)
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
        state::KalmanState,
        update_matrices::KalmanUpdate,
        external_input::Vector
)::KalmanState
    # Update the state vector
    pred_state_vec::Vector = (update_matrices.transition * state.state 
                              + update_matrices.control * external_input)
    pred_state_cov::Matrix = ((update_matrices.transition * state.cov 
                               * transpose(update_matrices.transition)) 
                               + update_matrices.noise)

    pred_state = KalmanState{eltype(pred_state_vec)}(pred_state_vec, pred_state_cov)
    return pred_state
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
        predicted_state::KalmanState,
        observation::KalmanObservation
)::KalmanState
    # Intermediate calculation (1)
    obs_transpose = transpose(observation.obs)

    # Compute the Kalman gain
    gain = (predicted_state.cov * obs_transpose
            * (observation.obs * predicted_state.cov * obs_transpose
               + observation.obsCov)^(-1))

    # Intermediate calculation (2)
    transformation = (I(size(predicted_state.cov)[1]) - gain * observation.obs)

    # Update the state estimate and covariance
    corr_state_vec = (predicted_state.state + gain
                      * (observation.meas - observation.obs * predicted_state.state))
    corr_cov = (transformation * predicted_state.cov * transpose(transformation)
                + gain * observation.obsCov * transpose(gain))
    corr_state = KalmanState{eltype(corr_state_vec)}(corr_state_vec, corr_cov)

    return corr_state
end;
