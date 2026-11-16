"""Scripted known-geometry controller. This is not a model rollout."""
import math

def opening_sequence():
    yield '1 / Hold the bottle with the right gripper',1,[.10,.18,1.02],[90,0,0],.09,90
    yield '1 / Align the right gripper with the bottle',1,[0,.12,.95],[90,0,0],.09,90
    yield '1 / Bring the fingers around the bottle',1,[0,-.012,.95],[90,0,0],.09,90
    yield '1 / Grip and stabilize the bottle body',1,[0,-.012,.95],[90,0,0],.050,25
    yield '2 / Reach the safety loop with the left gripper',0,[0,-.05,1.13],[180,0,0],.035,90
    yield '2 / Position the fingers around the loop',0,[0,-.049,1.066],[180,0,0],.035,25
    yield '2 / Pinch the safety loop',0,[0,-.049,1.066],[180,0,0],.003,15
    for deg in (15,30,50,70,90,110,130,150,170):
        t=math.radians(deg)
        yield '3 / Fold the safety loop down through contact',0,[0,-.0465-.0275*math.sin(t),1.040+.0275*math.cos(t)-.0015],[180,0,0],.003,10
    yield '3 / Release the lowered loop',0,[0,-.058,1.01],[180,0,0],.05,20
    yield '4 / Reposition the left gripper',0,[0,-.14,1.07],[180,0,0],.05,60
    yield '4 / Aim the fingertips at the lower button',0,[0,-.10,1.028],[-90,0,0],0,70
    yield '4 / Approach the rocker button',0,[0,-.053,1.028],[-90,0,0],0,35
    yield '5 / Press the rocker; the spring opens the lid',0,[0,-.043,1.028],[-90,0,0],0,40
