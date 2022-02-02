using LinearAlgebra
using Random

include("kalmanflt.jl")

function get_model_estimate(
    initial_state::Vector,
    measurement_time_sec)::Vector

    accel_ms2 = 9.8 # meter / sec / sec
    obs_mat = [1.0 measurement_time_sec; 0.0 1.0]
    external_input_vector = Vector([accel_ms2, accel_ms2])
    control_mat = [0.5 * measurement_time_sec ^ 2 0; 0 measurement_time_sec]
    model_estimate_vec = (obs_mat * initial_state
                          - (control_mat * external_input_vector))
    return model_estimate_vec
end

function get_measurement(
    initial_state::Vector,
    measurement_time_sec)::Vector
    """Get the next measurement
    In a 'real' physical system, we would not need to simulate the measurement"""

    model_estimate_vec = get_model_estimate(initial_state, measurement_time_sec)
    # Pseudo-measurement using classical physics calculation with noise
    position_noise_m = 5.0  # meter
    velocity_noise_ms = 0.1  # meter / sec
    artificial_noise_vec = Vector([position_noise_m * randn(1); velocity_noise_ms * randn(1)])
    measurement::Vector = model_estimate_vec + artificial_noise_vec
    return measurement

end

function constantaccel()

    accel_ms2 = 9.8  # meter / sec / sec
    delta_time_s = 0.1

    """Define a system where a falling object is released at rest at 1 km above the
    Earth's surface. 
    The altitude variance set to (10.0 meter)^2.
    The velocity variance set to (0.1 meter/sec)^2
    """
    initial_state_vector = Vector([1000.0, -0.001])
    initial_state_cov = Diagonal([10.0^2, 0.1^2])
    initial_state = kfstate{Float64}(initial_state_vector, initial_state_cov)

    """The system is just a particle falling due to gravity without air resistance"""
    state_transition_mat = [1.0 delta_time_s; 0.0 1.0]
    control_mat = [0.5 * delta_time_s^2 0; 0 delta_time_s]
    system_noise_mat = zeros(Float64, (2, 2))
    external_input_vector = -1 * Vector([accel_ms2, accel_ms2])
    system = kfsystem{Float64}(state_transition_mat, control_mat, system_noise_mat)

    """Initialize the observation matrix. The measurement and state is one-to-one"""
    obs_mat = Diagonal([1.0, 1.0])
    obsCov = Diagonal([5.0^2, 0.25^2])
    observations = kfobservation{Float64}(obs_mat, obsCov)

    next_state = kfstate{Float64}(initial_state.state, initial_state.cov)
    for step in 1:100
        # At current time, get measurement
        time_s = step * delta_time_s
        measurement_vec = get_measurement(initial_state.state, time_s)
        true_state_vec = get_model_estimate(initial_state.state, time_s)
        println("Time $time_s, measurement $measurement_vec, true $true_state_vec")

        # Predict the next state
        predicted_state = extrapulate_state(next_state, system, external_input_vector)

        # Update the estimate
        next_state = update_state(predicted_state, observations, measurement_vec)
        println("The Kalman gain state is $(next_state.state) with cov = $(next_state.cov)")

    end
end

constantaccel()