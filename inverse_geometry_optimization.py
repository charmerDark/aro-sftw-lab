#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated, jointlimitscost
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, COLLI_COST

from scipy.optimize import fmin_bfgs, fmin_slsqp


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None, disp=True):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    
    def cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        # get placement of LEFT_HAND and RIGHT_HAND
        oMleft_hand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMright_hand = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        
        # get placement of LEFT_HOOK and RIGHT_HOOK
        oMleft_hook = getcubeplacement(cube, LEFT_HOOK)
        oMright_hook = getcubeplacement(cube, RIGHT_HOOK)
        
        norm_diff = norm(pin.log(oMleft_hand.inverse() * oMleft_hook).vector) + \
                    norm(pin.log(oMright_hand.inverse() * oMright_hook).vector)
        collision_cost = COLLI_COST if collision(robot, q) else 0
        joint_cost = jointlimitscost(robot, q)
        
        return norm_diff + collision_cost + joint_cost
    
    def callback(q):
        pass
    
    qtarget = fmin_bfgs(cost, qcurrent, callback=callback, disp=disp)
    
    if not (collision(robot, qtarget) or jointlimitsviolated(robot, qtarget)):
        return qtarget, True
    else:
        return qcurrent, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)