# kalmanjl
 
 This is a Kalman filter implementation in Julia with examples.

 ## Requirements

 * ForwardDiff
 * SymPy

 ## Example: Falling Object Under Constant Acceleration

 A toy example of an object falling without air resistance is demonstrated in `constantaccel.ipynb` notebook. The state vector is (h, hdot) where `h -> h + v t + 1/2 g t^2` and `hdot -> hdot + g t`. The measurement is (h, hdot) and so the observation matrix is the identity matrix.

 ## Example: Falling Object With Terminal Velocity

 A toy example of an object falling air resistance is demonstrated in the `terminalvelocity.ipynb`. The state vector is (h, hdot) like the previous example, however the state update equation must be modified. The velocity is modeled as `v -> v_inf tanh(g t / v_inf)` where `v_inf` is the velocity as time approaches infinity. This example uses the Extended Kalman Filter.

 Instead of measuring the position and velocity, which makes the observation matrix the identity, the temperature is measured instead. A temperature versus altitude model is implemented to map to truth.

 An equivalent notebook using symbolic calculus is provided in `terminalvelocity_symbolic.ipynb`. However, note that the evaluation is very slow.