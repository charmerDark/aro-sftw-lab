#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
from pinocchio.utils import rotate
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import time
from inverse_geometry import computeqgrasppose
from tools import jointlimitsviolated, collision, COLLI_COST, jointlimitscost, projecttojointlimits, setcubeplacement
from solution import LMRREF

from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import pinv,inv,norm,svd,eig

from tools import setupwithmeshcat
    
robot, cube, viz = setupwithmeshcat()

discretisationsteps_newconf = 3 # To tweak later on
discretisationsteps_validedge = 3 # To tweak later on
k = 1000  # To tweak later on
delta_q = None # To tweak later on



def RAND_CONF():
    '''
    Return a random configuration, not in collision, with cube placement bound
    '''
    while True:
        # cube placement bound
        x_value = np.random.uniform(0.33, 0.40)
        y_value = np.random.uniform(-0.30, 0.11)
        z_value = np.random.uniform(0.93, 1.40)
        
        translation_array = np.array([x_value, y_value, z_value])
        random_placement = pin.SE3(rotate('z', 0), translation_array)
        setcubeplacement(robot, cube, random_placement)
        if not pin.computeCollisions(cube.collision_model, cube.collision_data, False):
            random_conf, success = computeqgrasppose(robot, robot.q0, cube, random_placement, viz = False)
            if success:
                return random_conf
            
def distance(q1, q2):    
    '''Return the euclidian distance between two configurations'''
    return np.linalg.norm(q2 - q1)
        
    
def NEAREST_VERTEX(G, q_rand):
    '''returns the index of the Node of G with the configuration closest to q_rand  '''
    min_dist = 10e4
    idx = -1
    for (i, node) in enumerate(G):
        dist = distance(node[1] , q_rand) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx


def ADD_EDGE_AND_VERTEX(G, parent, q):
    G += [(parent,q)]
    print('ADD_EDGE_AND_VERTEX: parent', parent)


def lerp_projecttoqgrasp(q0, q1, t):
    '''return interpolation configuration corresponds to a grasping pose'''  
    qt = q0 * (1 - t) + q1 * t
    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    
    def cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        # get placement of LEFT_HAND and RIGHT_HAND
        oMleft_hand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMright_hand = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        
        leftMright = oMleft_hand.inverse() * oMright_hand
        
        # calculate the difference between leftMright and LMRREF
        norm_diff = norm(pin.log(leftMright.inverse() * LMRREF).vector)
        collision_cost = COLLI_COST if collision(robot, q) else 0
        joint_cost = jointlimitscost(robot, q)
        
        return norm_diff + collision_cost + joint_cost
    
    def callback(q):
        pass
    
    qgrasp = fmin_bfgs(cost, qt, callback=callback, disp=False)
    
#     print(collision(robot, q0), collision(robot, q1), collision(robot, qt), collision(robot, qgrasp))
    if not (collision(robot, qgrasp) or jointlimitsviolated(robot, qgrasp)):
        return qgrasp, True
    else:
        return qt, False
    

def path_qgrasp(qstart, qend, discretisationsteps):
    '''return path from qstart to qend corresponds to a grasping pose, where assume that qstart and qend already in a grasping pose'''
    path = [qstart]
    dt = 1 / discretisationsteps
    for i in range(1, discretisationsteps):
        path += [lerp_projecttoqgrasp(qstart, qend, dt * i)[0]]
    path += [qend]
    return path


def NEW_CONF(q_near, q_rand, discretisationsteps, delta_q = None):
    '''return the closest configuration q_new such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta_q'''
    q_end = q_rand.copy()
    dist = distance(q_near, q_rand)
    if delta_q is not None and dist > delta_q:
        #compute the configuration that corresponds to a path of length delta_q
        q_end = lerp(q_near, q_rand, delta_q / dist)
        # now dist == delta_q
    dt = 1 / discretisationsteps
    for i in range(1, discretisationsteps):
        q, success = lerp_projecttoqgrasp(q_near, q_end, dt * i)
        if not success:
            return lerp_projecttoqgrasp(q_near, q_end, dt * (i-1))[0], i-1
    return q_end, discretisationsteps


def VALID_EDGE(q_new, q_goal, discretisationsteps):
    return np.linalg.norm(q_goal - NEW_CONF(q_new, q_goal, discretisationsteps)[0]) < 1e-3


def rrt(q_init, q_goal, k, delta_q):
    G = [(None, q_init)]
    for i in range(k):
        q_rand = RAND_CONF()
        q_near_index = NEAREST_VERTEX(G, q_rand)
        q_near = G[q_near_index][1]
        q_new, steps= NEW_CONF(q_near, q_rand, discretisationsteps_newconf, delta_q)
        
        # if q_new == q_near, skip
        print(i, 'steps =', steps)
        if steps == 0:
            continue
            
        ADD_EDGE_AND_VERTEX(G, q_near_index, q_new)
        if VALID_EDGE(q_new, q_goal, discretisationsteps_validedge):
            print ("path found!")
            ADD_EDGE_AND_VERTEX(G, len(G)-1, q_goal)
            return G, True
    print("path not found")
    return G, False


def getpath(G):
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [node[1]] + path
        node = G[node[0]]
    path = [G[0][1]] + path
    
    extend_steps = 10
    path_extend = path[0: 1]
    for q0, q1 in zip(path[:-1],path[1:]):
        path_extend += path_qgrasp(q0, q1, extend_steps)[1:]
    return path_extend


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    G, foundpath = rrt(qinit, qgoal, k, delta_q)
    
    return foundpath and getpath(G) or []


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot, path, dt=0.5, viz=viz) #you ll probably want to lower dt
    
