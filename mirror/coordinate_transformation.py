# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:40:39 2023

@author: ki52soho
"""

import numpy as np
import sympy as sy


class CoordinateTransform():
    
    def __init__(self, d, D, rotation_degree):
        # Mirror
        self.d = d # distance center of rotation to mirror surface, mm
        self.D = D  # Distance mirror center (undeflected) to target plane, mm
        self.rotation_degree = rotation_degree
        
        
        self.r_C = np.array([0, 0, self.d]) # center of rotation, origin
        
        
        # incoming beam in yz plane
        self.n_0 = self.normalize(np.array([0,-1,1]))
        self.r_OP_0 = np.array([0,1, -1])
        
        # target plane
        
        
        alpha = np.deg2rad(self.rotation_degree)
        self.A_TI = np.array([[ 1, 0 ,0], 
                        [ 0, np.cos(alpha), -np.sin(alpha)], 
                        [ 0, np.sin(alpha), np.cos(alpha)]])
        self.A_IT = np.transpose(self.A_TI)
        self.n_t = np.dot(self.A_IT, np.array([0,0,1]))
        self.r_OT = np.dot(self.A_IT, np.array([0,0,-D]))
        
    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    def getXYFromNormalVector(self, n, d, D):
        r_C = np.array([0, 0, d]) # center of rotation
        # xy coordinate system defined with respect to 90° incidence angle
        n_0 = self.normalize(np.array([0,0,1]))
        r_OP_0 = np.array([0,0, -1])
    
        # target plane
        n_t = np.array([0,0,1])
        r_OT = np.array([0,0,-D]) 
        
        scaling = D*np.tan(np.deg2rad(50))
        
        r_OP_2 = self.getSpotOnTargetPlane(n, r_C, d, n_0, r_OP_0, r_OT, n_t)
        x = r_OP_2[0]/scaling
        y = r_OP_2[1]/scaling
        return x,y
    
    def getNormalVectorFromXY(self, x,y, d):
        #distance d between center of rotation and mirror surface (mm)
        if d==0:
            return self.getNormalVectorFromXY_d0(x,y) # faster
        
        # Parameters as used in OQC calibration
        r_C = np.array([0, 0, d]) # center of rotation
        n_0 = np.array([0,0,1])
        r_OP_0 = np.array([0,0, -1])
    
        # target plane
        D = 90 # Distance center of rotation to target plane, mm
        n_t = np.array([0,0,1])
        r_OT = np.array([0,0,-D])
        
        n_m1, n_m2 = sy.symbols("n_m1 n_m2")
        n_m = sy.matrices.Matrix([n_m1, n_m2, -sy.sqrt(1-(n_m1**2+n_m2**2))])
        r_OP_2 = self.sy_getSpotOnTargetPlane(n_m, r_C, d, n_0, r_OP_0, r_OT, n_t)
        
        A_TI = np.diag([1,1,1])
        I_r_TP_2 = r_OP_2 - sy.matrices.Matrix(r_OT)
        T_r_TP_2 = np.dot(A_TI, I_r_TP_2)
        T_r_TP_2 = sy.simplify(T_r_TP_2)
        
        
        scaling = D*np.tan(50/180*np.pi)
        res = sy.solvers.solvers.nsolve((T_r_TP_2[0][0]-x*scaling,
                                         T_r_TP_2[1][0]-y*scaling),
                                        (n_m1, n_m2), (0, 0))
        n_x = float(res[0])
        n_y = float(res[1])
        
        n = np.array([n_x, n_y, -np.sqrt(1-(n_x**2+n_y**2))])
        
        return n
    
    def getNormalVectorFromXY_d0(self, x,y): 
        #approximation for centered beam and d -> 0
        n0 = np.array([0,0, 1]) # xy coordinate system defined with respect to 90° incidence angle
        
        r = self.normalize(np.array([x, y, -1/np.tan(np.deg2rad(50))])) # direction of reflected beam
        return self.normalize(r - n0)
    
    def getNormalVectorFromTargetXY_d0(self, x_t, y_t, A_IT, D, n0):
        #approximation for centered beam and d -> 0
        n_t = np.dot(A_IT, np.array([0,0,1]))
        r_OT = np.dot(A_IT, np.array([0,0,-D]))
        T_r_TP_2 =  np.array([x_t, y_t, 0])
        I_r_TP_2 =  np.dot(A_IT, T_r_TP_2)
        r_OP_2 = r_OT + I_r_TP_2
        n_1 = self.normalize(r_OP_2)
        
        n_m = self.normalize(n_1-n0)
        return n_m
    
    def getSpotOnTargetPlane(self, n_m, r_C, d, n_0, r_OP_0, r_OT, n_t):
        #r_M = r_C + d * n_m # center of mirror surface
        
        # intersection of incoming beam with mirror (beam clipping not checked)
        t_1 = (np.dot(r_C-r_OP_0, n_m)+d) / np.dot(n_0, n_m)
        r_OP_1 = r_OP_0 + t_1 * n_0 
        
        n_1 = n_0 - 2*np.dot(n_0, n_m)*n_m # reflected beam
        
        # intersection of reflected beam with target plane (beam clipping not checked)
        # no check of whether the intersection happens in the correct half-space (t2 > 0)
        t_2 = np.dot(r_OT-r_OP_1, n_t) / np.dot(n_1, n_t)
        if t_2 < 0:
            print('Warning: intersection in wrong half-space')
        r_OP_2 = r_OP_1 + t_2 * n_1
        return r_OP_2
    
    def sy_getSpotOnTargetPlane(self, n_m, r_C, d, n_0, r_OP_0, r_OT, n_t):
        # same as above but for symbolic calculations
        #r_M = r_C + d * n_m # center of mirror surface
        n_m = sy.matrices.Matrix(n_m)
        r_C = sy.matrices.Matrix(r_C)
        n_0 = sy.matrices.Matrix(n_0)
        r_OP_0 = sy.matrices.Matrix(r_OP_0)
        r_OT = sy.matrices.Matrix(r_OT)
        n_t = sy.matrices.Matrix(n_t)
        
        t_1 = (n_m.dot(r_C-r_OP_0)+d) / n_0.dot(n_m)
        r_OP_1 = r_OP_0 + t_1 * n_0 # center point of incoming beam on mirror
        
        n_1 = n_0 - 2*n_0.dot(n_m)*n_m # reflected beam
        
        t_2 = n_t.dot(r_OT-r_OP_1) / n_1.dot(n_t)
        r_OP_2 = r_OP_1 + t_2 *n_1
        return r_OP_2    
    
     
    
    def target_to_mirror(self, x_target, y_target):
        x_input = [] # to be calculated mirror coordinates
        y_input = [] # to be calculated mirror coordinates
        
        if self.d==0 and abs(np.dot(self.r_OP_0, self.n_0)/np.linalg.norm(self.r_OP_0) + 1) < 1e-9:
            # calculation is simple if d is 0 and incoming beam hits mirror in the center (origin)
            simpleMode = True
            # print("Using simple mode")
        else:
            simpleMode = False
            n_m1, n_m2 = sy.symbols("n_m1 n_m2")
            n_m = sy.matrices.Matrix([n_m1, n_m2, -sy.sqrt(1-(n_m1**2+n_m2**2))])
            r_OP_2 = self.sy_getSpotOnTargetPlane(n_m, self.r_C, self.d, self.n_0, self.r_OP_0, self.r_OT, self.n_t)
            I_r_TP_2 = r_OP_2 - sy.matrices.Matrix(self.r_OT)
            T_r_TP_2 = np.dot(self.A_TI, I_r_TP_2)
            T_r_TP_2 = sy.simplify(T_r_TP_2)
            # print("Not using simple mode")
        
            
            
        
        progress = 0
        n_target = len(x_target)
        
        for i, (x_t, y_t) in enumerate(zip(x_target, y_target)):
            if simpleMode:
                n = self.getNormalVectorFromTargetXY_d0(x_t, y_t, self.A_IT, self.D, self.n_0)
                x,y = self.getXYFromNormalVector(n, 0, 90) # D is arbitrary for d==0
            
            else:
                res = sy.solvers.solvers.nsolve((T_r_TP_2[0][0]-x_t,
                                                 T_r_TP_2[1][0]-y_t),
                                                (n_m1, n_m2), (0, 0))
                n_x = float(res[0])
                n_y = float(res[1])
        
                n = np.array([n_x, n_y, -np.sqrt(1-n_x**2-n_y**2)])
                x,y = self.getXYFromNormalVector(n, self.d, 90)
                
            x_input.append(x)
            y_input.append(y)
            
            if (i+1)/n_target> progress+0.1:
                progress = i/n_target
                print(int(round(progress,1)*100), '%')
        
        # get indices for feasible points:
        i_feasible = []    
        # discard grid-points outside unit circle
        for i, (x,y) in enumerate(zip(x_input, y_input)):
            if x**2+y**2 <= 1:
                i_feasible.append(i)
        
        x_input = np.array(x_input)
        y_input = np.array(y_input)
    
        
        return x_input[i_feasible], y_input[i_feasible]


