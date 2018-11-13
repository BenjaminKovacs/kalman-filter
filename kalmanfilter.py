import matplotlib.pyplot as plt
import numpy as np

# copied from https://github.com/rblilja/AltitudeKF/blob/master/altitude_kf.h

global h
global v
global P

h = 0;
v = 0;

P = [[1,0],[0,1]]

global Q_accel
global R_altitude

# need to be changed later
# @param Q_accel covariance of acceleration input signal (σ^2).
# @param R_altitude covariance of the altitude measurement (σ^2).
Q_accel = 1
R_altitude = 1

def propagate(acceleration, dt):
    global h
    global v
    global P
    
    global Q_accel
    global R_altitude
    
    _dtdt = dt * dt;
    
    # The state vector is defined as x = [h v]' where  'h' is altitude above ground and 'v' velocity, both
    # aligned with the vertical direction of the Earth NED frame, but positive direction being upwards to zenith.
    
    # State-space system model 'x_k = A*x_k-1 + B*u_k is given by:
    #
    #	x_k = [ h_k ] = [ 1 dT ] * [ h_k-1 ] + [ 1/2*dT^2 ] * u_k
    #  	      [ v_k ]   [ 0  1 ]   [ v_k-1 ]   [ dT       ]
    #
    #			   A			     B
    #
    # where 'u_k' is our acceleration input signal.
    
    # Propagation of the state (equation of motion) by Euler integration
    h = h + v*dt + 0.5*acceleration*_dtdt;
    v = v + acceleration*dt;

    # The "a priori" state estimate error covariance 'P_k|k-1 = A * P_k-1 * A' + Q_k' is calculated as follows:
    #
    # P_k|k-1 = [ 1 dT ] * P_k-1 * [  1 0 ] + Q_k
    #	     [ 0  1 ]	        [ dT 1 ]

    # The process noise covariance matrix 'Q' is a bit trickier to derive, but consider some additive noise 'w_k' perturbing the
    # true acceleration 'a_k', thus the input signal is 'u_k = a_k + w_k'. The affect of 'w_k' on the state estimate is by linearity
    # described by [1/2*dT^2 dT]' i.e. the input matrix 'B'. We call this new matrix 'G'.
    #
    # Then, by definition* 'Q' equals 'G * G' * σ^2', which in our case translates into:
    #
    # Q_k = G_k * G'_k * σ_accelerometer^2 = [(dT^4)/4 (dT^3)/2] * σ_accelerometer^2
    #					  [(dT^3)/2     dT^2]
    #
    # * I only get half of the math showing 'Q = G * G' * σ^2', so I hide myself behind 'by definition'.

    # Calculate the state estimate covariance
    #
    # Repeated arithmetics
    _Q_accel_dtdt = Q_accel * _dtdt;
    #
    P[0][0] = P[0][0] + (P[1][0] + P[0][1] + (P[1][1] + 0.25*_Q_accel_dtdt) * dt) * dt;
    P[0][1] = P[0][1] + (P[1][1] + 0.5*_Q_accel_dtdt) * dt;
    P[1][0] = P[1][0] + (P[1][1] + 0.5*_Q_accel_dtdt) * dt;
    P[1][1] = P[1][1] + _Q_accel_dtdt;
    
def update(altitude):
    global h
    global v
    global P
    
    global Q_accel
    global R_altitude
    
    # Observation vector 'zhat' from the current state estimate:
    #
    # zhat_k = [ 1 0 ] * [ h_k ]
    #                    [ v_k ]
    #             H

    # 'H' is constant, so its time instance I'm using below is a bit ambitious.

    # The innovation (or residual) is given by 'y = z - zhat', where 'z' is the actual observation i.e. measured state.

    # Calculate innovation, in this particular case we observe the altitude state directly by an altitude measurement
    y = altitude - h;

    # The innovation covariance is defined as 'S_k = H_k * P_k|k-1 * H'_k + R_k', for this particular case
    # 'H_k * P_k|k-1 * H'_k' is equal to the first row first column element of 'P_k|k-1' i.e. P_00.

    # The Kalman gain equals 'K_k = P_k|k-1 * H'_k * S_k^-1', where
    #
    # P_k|k-1 * H'_k = [ P_00 ]
    #                  [ P_10 ]
    #
    # and 'S_k^-1' equals '1/S_k' since 'S_k^-1' is being a scalar (that is a good thing!).

    # Calculate the inverse of the innovation covariance
    Sinv = 1.0 / (P[0][0] + R_altitude);

    # Calculate the Kalman gain
    K = [P[0][0] * Sinv, P[1][0] * Sinv];

    # Update the state estimate
    h += K[0] * y;
    v += K[1] * y;

    # The "a posteriori" state estimate error covariance equals 'P_k|k = (I - K_k * H_k) * P_k|k-1', where
    #
    #  (I - K_k * H_k) = ( [ 1 0 ] - [ K_0 ] * [ 1 0 ] ) = [ (1-K_0) 0  ] , thus
    #                    ( [ 0 1 ]   [ K_1 ]           )   [ -K_1    1  ]
    #
    #  P_k|k = (I - K_k * H_k) * P_k+1|k = [ (1-K_0) 0 ] * [ P_00 P_01 ] = [ (1-K_0)*P_00       (1-K_0)*P_01       ]
    #					[ -K_1    1 ]   [ P_10 P_11 ]   [ (-K_1*P_00 + P_10) (-K_1*P_01 + P_11) ]

    # Calculate the state estimate covariance
    P[0][0] = P[0][0] - K[0] * P[0][0];
    P[0][1] = P[0][1] - K[0] * P[0][1];
    P[1][0] = P[1][0] - (K[1] * P[0][0]);
    P[1][1] = P[1][1] - (K[1] * P[0][1]);

# [(time, value)]    
fakeAccelerometerData = []
fakeAltimeterData = []
testH = 0
for i in range(50):
    fakeAccelerometerData.append((i,5))
    testV = 5*i
    testH += testV
    fakeAltimeterData.append((i,testH))
    

#for i in range(25):
#    fakeAltimeterData.append((2*i,i**2))

accelCount = 1
altCount = 0
while accelCount<len(fakeAccelerometerData) or altCount<len(fakeAltimeterData):
    print(h,v)
    if altCount>=len(fakeAltimeterData) or fakeAccelerometerData[accelCount][0] < fakeAltimeterData[altCount][0]:
        dt = fakeAccelerometerData[accelCount][0] - fakeAccelerometerData[accelCount-1][0]
        propagate(fakeAccelerometerData[accelCount][1], dt)
        accelCount+=1
    else:
        update(fakeAltimeterData[altCount][1])
        altCount+=1
        
print('actual',testH, testV)


