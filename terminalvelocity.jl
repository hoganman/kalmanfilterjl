using LinearAlgebra
using Random

include("kalmanflt.jl")

function get_velocity_infinity(temperature_K)
    """v_inf between ~46 and ~54 meters / sec"""
    return -0.1481 * temperature_K + 94.44
end

function get_temperature(altitude_m, vary_slope_by=0, vary_intercept_by=0)
    """"""
    slope = -0.006_835 # Kelvin / meter
    intercept = 288.706 # Kelvin
    if vary_slope_by > 0
        slope *= vary_slope_by
    end
    if vary_intercept_by > 0
        intercept *= vary_intercept_by
    end
    temerature = slope * altitude_m + intercept
    return temerature
end

function get_observation_mat(
    altitude_m,
    vary_temperature_slope_by=0,
    vary_temperature_intercept_by=0)::Matrix
    temperature_K = get_temperature(altitude_m, 
                                  vary_temperature_slope_by, 
                                  vary_temperature_intercept_by)
    observation_mat = [temperature_K/altitude_m 0.0]
    return observation_mat
end

function get_control_mat(
    state_vec::Vector,
    total_time_sec,
    delta_time_sec)::Matrix

    accel_ms2 = -9.8 # meter / sec^2
    temperature_K = get_temperature(state_vec[1])
    vel_inf_ms = get_velocity_infinity(temperature_K)
    arg = accel_ms2 * total_time_sec / vel_inf_ms
    current_acceleration_ms2 = accel_ms2 * sech(arg)^2
    current_jerk_ms3 = -2 * accel_ms2^2 / vel_inf_ms * sech(arg)^2 * tanh(arg)
    control_mat11 = (1. / factorial(2) * current_acceleration_ms2 * delta_time_sec^2
                     + 1. / factorial(3) * current_jerk_ms3 * delta_time_sec^3)
    control_mat_12 = 0
    control_mat_21 = 0
    control_mat_22 = -(current_acceleration_ms2 * delta_time_sec
                       + 1. / factorial(2) * current_jerk_ms3 * delta_time_sec^2)
    control_mat = [control_mat11 control_mat_12; control_mat_21 control_mat_22]
    return control_mat
end

function get_model_estimate(
    state_vec::Vector,
    total_time_sec,
    delta_time_sec)::Vector

    state_transition = [1.0 -delta_time_sec; 0.0 1.0]
    control_mat = get_control_mat(state_vec, total_time_sec, delta_time_sec)
    external_input_vector = Vector([1, 1])
    model_estimate_vec = (state_transition * state_vec
                          + control_mat * external_input_vector)
    return model_estimate_vec
end

function get_measurement(
    true_state::Vector,
    vary_temperature_slope_by=0,
    vary_temperature_intercept_by=0)::Vector
    """Get the next measurement
    In a 'real' physical system, we would not need to simulate the measurement"""

    altitude_m = true_state[1]
    observation_mat = get_observation_mat(altitude_m, 
                                          vary_temperature_slope_by, 
                                          vary_temperature_intercept_by)

    artificial_noise_K = 0.1 * randn(1) # Kelvin
    measurement::Vector = observation_mat * true_state + artificial_noise_K
    return measurement

end

function terminalvelocity()

    delta_time_s = 0.1

    """Use these two variables to simulate differences variations between the truth and observation
    for the temperature vs altitude
    """
    temperature_variation_slope = 0.96
    temperature_variation_intercept = 1.04

    """Define a system where a falling object is thought to be released at rest 
    at 3 km above the Earth's surface with wind resistence
    """
    true_state_vector = Vector([3020.0, +0.05])
    initial_state_vector = Vector([3000.0, -0.001])
    initial_state_cov = Diagonal([1000.0^2, 10.0^2])
    initial_state = kfstate{Float64}(initial_state_vector, initial_state_cov)

    state_transition_mat = [1.0 -delta_time_s; 0.0 1.0]
    control_mat = get_control_mat(initial_state_vector, delta_time_s, delta_time_s)
    system_noise_mat = [10.0^2 0; 0 1.0^2]
    external_input_vector = Vector([1, 1])
    system = kfsystem{Float64}(state_transition_mat, control_mat, system_noise_mat)

    """Initialize the observation matrix"""
    obs_mat = get_observation_mat(initial_state.state[1], temperature_variation_slope, temperature_variation_intercept)
    obsCov = 5.0^2 * I(1)  # Kelvin^2
    observations = kfobservation{Float64}(obs_mat, obsCov)

    next_state = kfstate{Float64}(initial_state.state, initial_state.cov)
    for step in 1:100

        # At current time, get measurement
        time_s = step * delta_time_s
        true_state_vector = get_model_estimate(true_state_vector, time_s, delta_time_s)
        measurement_vec = get_measurement(true_state_vector, temperature_variation_slope, temperature_variation_intercept)

        # Predict the next state
        predicted_state = extrapulate_state(next_state, system, external_input_vector)

        # Update the estimate
        next_state = update_state(predicted_state, observations, measurement_vec)

        # Update the system control and observation matrix for the next measurement
        new_control_mat = get_control_mat(next_state.state, time_s, delta_time_s)
        system.control = new_control_mat
        new_observations_mat = get_observation_mat(next_state.state[1], temperature_variation_slope, temperature_variation_intercept)
        observations.obs = new_observations_mat

        println("The Kalman gain state is $(next_state.state)")
        println("The true state is $(true_state_vector)")

    end

end


terminalvelocity()
