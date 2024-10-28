#!/usr/bin/env python

import sys
import numpy as np

# functions
def rnd(n): return np.round(n, 5)
def rad(d): return (d*np.pi)/180 # rad from deg
def deg(r): return (r*180)/np.pi # deg from rad
def rot(l,n): return l[n:]+l[:n] # rotate list
def trans(a): # transpose last 2 axis
    s=list(range(len(a.shape)))
    return np.transpose(a,s[:-2]+s[-1:-3:-1])
def Rz(a):
    c,s=np.cos(a),np.sin(a)
    z,u=1.0*(a!=a),1.0*(a==a)
    r=np.array([[c,s,z],[-s,c,z],[z,z,u]])
    a=rot(list(range(len(r.shape))),2)
    return np.transpose(r,a)
def Rx(a):
    c,s=np.cos(a),np.sin(a)
    z,u=1.0*(a!=a),1.0*(a==a)
    r=np.array([[u,z,z],[z,c,s],[z,-s,c]])
    a=rot(list(range(len(r.shape))),2)
    return np.transpose(r,a)
def Rmiller(n,b):
    c=np.cross(n,b,0,0)
    r=[v/(v**2).sum(axis=0)**0.5 for v in [b,c,n]]
    return trans(r)
def angle(r): return np.arccos((np.trace(r,axis1=-2,axis2=-1)-1)/2.)
def axis(r):
    r=0.5*(r-trans(r))
    a=rot(list(range(len(r.shape))),-2)
    r=np.transpose(r,a)
    return np.transpose(np.array([r[1,2],r[2,0],r[0,1]]))
def miller(r):
    return r[:,2],r[:,0]

# tests
def test_Rzx():
    rz,rx=Rz(np.pi/2),Rx(np.pi/2)
    assert(np.isclose(np.identity(3),rz@np.transpose(rz)).all())
    assert(np.isclose(np.identity(3),rx@np.transpose(rx)).all())
    assert(np.isclose(np.array([[0,1,0],[-1,0,0],[0,0,1]]),rz).all())
    assert(np.isclose(np.array([[1,0,0],[0,0,1],[0,-1,0]]),rx).all())
test_Rzx()

# higher-level functions
def print_miller(r):
    n,b=miller(r)
    print("miller n (plane):\n",rnd(n))
    print("miller b (direction):\n",rnd(b))
def print_axis_angle(r):
    a,A=angle(r),axis(r)
    print("angle(rad):\n",a)
    print("angle(deg):\n",rnd(deg(a)))
    print("axis:\n",rnd(A))

# parse command line arguments
if len(sys.argv)==2:
    try:
        a=np.loadtxt(sys.argv[1])
    except ValueError:
        a=np.loadtxt(sys.argv[1],skiprows=1)
    a=np.unstack(a,axis=1)
else:
    a=[float(ai) for ai in sys.argv[1:]]

if len(a)==3:

    # Euler angles
    ea=[rad(ai) for ai in a] # radians
    if isinstance(ea[0],np.ndarray):
        print("EA(deg):\n",np.transpose(np.array(a)))
        print("EA(rad):\n",np.transpose(np.array(ea)))
    else:
        print("EA(deg):",a)
        print("EA(rad):",ea)

    # Rotation matrices
    phi1,Phi,phi2=ea
    Rphi1,RPhi,Rphi2=Rz(phi1),Rx(Phi),Rz(phi2)
    print("Rphi1:\n", rnd(Rphi1))
    print("RPhi:\n", rnd(RPhi))
    print("Rphi2:\n", rnd(Rphi2))

    # Total rotation matrix
    R=Rphi1@RPhi@Rphi2
    print("R:\n", rnd(R))

    # Miller indices
    print_miller(R)

    # Axis and angle
    print_axis_angle(R)

elif len(a)==6:

    # plane normal and direction
    h,k,l,u,v,w=a
    n,b=np.array([h,k,l]),np.array([u,v,w])
    print('miller n (plane):',n)
    print('miller b (direction):',b)

    # Rotation matrix
    R=Rmiller(n,b)
    print("R:\n", rnd(R))

    # Axis and angle
    print_axis_angle(R)

else:

    sys.stderr.write(sys.argv[0]+' filename.txt  calculate rotation from text file\n')
    sys.stderr.write(sys.argv[0]+' phi1 Phi phi2 calculate rotation for Euler angles (phi1,Phi,phi2)\n')
    sys.stderr.write(sys.argv[0]+' h k l u v w   calculate rotation for Miller indices {hkl}<uvw>\n')
    exit(0)
