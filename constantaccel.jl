using LinearAlgebra
using Random

include("kalmanflt.jl")

function get_measurement(
    initial_state::Vector,
    measurement_time_sec)::Vector

    accel_ms2 = 9.8 # meter / sec / sec
    # Pseudo-measurement using classical physics calculation with noise
    position_noise_m = 7.5  # meter
    velocity_noise_ms = 0.15  # meter / sec
    noise_vector = Vector([position_noise_m * randn(1); velocity_noise_ms * randn(1)])
    measurement::Vector = ([1.0 measurement_time_sec; 0.0 1.0] * initial_state
                           - accel_ms2 * ([0.5 * measurement_time_sec^2 0; 0 measurement_time_sec] 
                           * Vector([1., 1.]))) + noise_vector
    return measurement

end

function constantaccel()

    input_accel_ms2 = 9.8  # meter / sec / sec
    delta_time_s = 0.1

    """Define a system where a falling object is released at rest at 1 km above the
    Earth's surface. 
    The altitude variance set to (10.0 meter)^2.
    The velocity variance set to (0.1 meter/sec)^2
    """
    initial_state_vector = Vector([1000.0, 0.0])
    initial_state_cov = Diagonal([10.0^2, 0.1^2])
    initial_state = klstate{Float64}(initial_state_vector, initial_state_cov)

    """The system is just a particle falling due to gravity with air resistance"""
    state_transition_mat = [1.0 delta_time_s; 0.0 1.0]
    control_mat = [0.5 * delta_time_s ^ 2 0; 0 delta_time_s]
    external_input_vector = Vector([input_accel_ms2, input_accel_ms2])
    system_noise_mat = zeros(Float64, (2, 2))
    system = klsystem{Float64}(state_transition_mat, control_mat, system_noise_mat)

    for step in 1:10
        time_s = step * delta_time_s
        measurement_vec = get_measurement(initial_state.state, time_s)
        println("Time $time_s, measurement $measurement_vec")
    end
end

constantaccel()