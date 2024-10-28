#!/usr/bin/env python

import sys
import numpy as np

# functions
def rnd(n): return np.round(n, 5)
def rad(d): return (d*np.pi)/180 # rad from deg
def deg(r): return (r*180)/np.pi # deg from rad
def Rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])
def Rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])
def angle(r): return np.arccos((np.trace(r)-1)/2.)
def axis(r):
    r=0.5*(r-np.transpose(r))
    return np.array([r[1,2],r[2,0],r[0,1]])

# tests
def test_Rzx():
    rz,rx=Rz(np.pi/2),Rx(np.pi/2)
    assert(np.isclose(np.identity(3),rz@np.transpose(rz)).all())
    assert(np.isclose(np.identity(3),rx@np.transpose(rx)).all())
    assert(np.isclose(np.array([[0,1,0],[-1,0,0],[0,0,1]]),rz).all())
    assert(np.isclose(np.array([[1,0,0],[0,0,1],[0,-1,0]]),rx).all())
test_Rzx()

# higher-level functions
def print_axis_angle(r):
    a,A=angle(r),axis(r)
    print('angle(rad):',a)
    print('angle(deg):',rnd(deg(a)))
    print('axis:',rnd(A))

if len(sys.argv)==4:

    # Euler angles
    ea=[float(a) for a in sys.argv[1:]]  # degrees
    print("EA(deg):",ea)
    ear=[rad(a) for a in ea] # radians
    print("EA(rad):",ear)

    # Rotation matrices
    phi1,Phi,phi2=ear
    Rphi1,RPhi,Rphi2=Rz(phi1),Rx(Phi),Rz(phi2)
    print("Rphi1:\n", rnd(Rphi1))
    print("RPhi:\n", rnd(RPhi))
    print("Rphi2:\n", rnd(Rphi2))

    # Total rotation matrix
    R=Rphi1@RPhi@Rphi2
    print("R:\n", rnd(R))

    # Axis and angle
    print_axis_angle(R)

else:

    sys.stderr.write(sys.argv[0]+' phi1 Phi phi2 calculate rotation for Euler angles (phi1,Phi,phi2)\n')
    exit(0)
