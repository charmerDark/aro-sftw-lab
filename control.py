#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    
    q_target = trajs[0](tcurrent)
    vq_target = trajs[1](tcurrent)
    vvq_target = trajs[2](tcurrent)
    
    M = pin.crba(robot.model, robot.data, q)
    nle = robot.data.nle.reshape((robot.model.nq, 1))

    vvq_d =  vvq_target - Kp * (q - q_target) - Kv * (vq - vq_target)

    torques = np.dot(M, vvq_d).flatten() + nle.flatten()
    sim.step(torques)


if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    
    # setting initial configuration
    sim.setqsim(q0)
    
    # get path from q0 to qe
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    print("#######")
    print(len(path)) 
    print("#######")
    def maketraj(path, T):

        modified_path = [path[0]] * 3 + path + [path[-1] ] * 3
        print(len(modified_path))
        q_of_t = Bezier(modified_path, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t
    
    
    total_time = len(path) / 5
    trajs = maketraj(path, total_time)
    
    tcur = 0.
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    