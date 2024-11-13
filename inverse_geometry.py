#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, collision, jointlimitsviolated


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    """
    Function computes a final position that can match goal. Not path to it
    """
    setcubeplacement(robot, cube, cubetarget)
    q = qcurrent.copy()
    DT = 1e-2
    pin.framesForwardKinematics(robot.model,robot.data,q)

    # Loop on an inverse kinematics for 1000 iterations.
    for i in range(1000):  # Integrate over 10 second of robot life

        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)
        
        oMcubeL = getcubeplacement(cube, LEFT_HOOK) 
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
        
        #Right hand task
        oMrarm = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        o_Jrarm = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND))
        o_right = pin.log(oMrarm.inverse() * oMcubeR).vector
        
        
        #Left hand task
        oMlarm = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        o_Jlarm = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND))
        o_left = pin.log(oMlarm.inverse() * oMcubeL).vector
        
        vq =  pinv(o_Jlarm) @ o_left + pinv(o_Jrarm) @ o_right # could do a single vetor packing larm and rarm params

        
        q = pin.integrate(robot.model, q, vq * DT)
    if not (collision(robot, q) or jointlimitsviolated(robot, q)):
        return q, True
    
    else:
        return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
