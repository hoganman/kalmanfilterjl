# kalmanjl
 
 This is a simple Kalman filter implementation in Julia.

 ## Requirements

 Julia 1.7 or at least Julia supporting the LinearAlgebra package.

 ## Example: Falling Object Under Constant Acceleration

 A toy example of an object falling without air resistance is implemented in `constantaccel.jl`. The state vector is (h, hdot) where `h -> h + v t + 1/2 g t^2` and `hdot -> hdot + g t`. The measurement is (h, hdot) and so the observation matrix is the identity matrix.

 ## Example: Falling Object Approaching Terminal Velocity

 A toy example of an object falling air resistance is implemented in `terminalvelocity.jl`. The state vector is (h, hdot) like the previous example, however the state update equation must be modified. The velocity is modeled as `v -> v_inf tanh(g t / v_inf)` where `v_inf` is the velocity as time approaches infinity. The first and second deriviative of this function are used to update the state vector `h -> h + v t + 1/2 vdot t^2 + 1/3! vdotdot t^3` and `hdot -> hdot + vdot t + 1/2 vdotdot t^2`. The third-order derivatives are not included since they are very small comparitively.

 Instead of measuring the position and velocity, which makes the observation matrix the identity, the temperature is measured instead. A temperature versus altitude model is implemented to map to truth.