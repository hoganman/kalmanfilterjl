using LinearAlgebra


mutable struct klfilter{T}

    # Observation matrix
    observation::Matrix{T}
    # Uncertainty on observation
    cov::Matrix{T}
    # Kalman gain
    gain::Matrix{T}

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