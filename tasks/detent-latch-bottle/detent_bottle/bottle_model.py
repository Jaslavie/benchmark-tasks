"""Procedural bottle geometry; dimensions and materials are estimates."""
import math

def scene_xml():
    # Visual shells are separate from convex mechanical contact surfaces.
    meshes = []
    def ring(name, radius, z, width, height, material, n=96, center_y=0, slope=0):
        vertices, faces = [], []
        for r, h in [(radius+width, -height), (radius+width, height),
                     (radius-width, height), (radius-width, -height)]:
            for i in range(n):
                a = 2*math.pi*i/n
                vertices.extend([r*math.cos(a), r*math.sin(a), h-slope*math.sin(a) if h > 0 else h])
        for j in range(4):
            for i in range(n):
                a, b = j*n+i, j*n+(i+1)%n
                c, d = ((j+1)%4)*n+(i+1)%n, ((j+1)%4)*n+i
                faces.extend([a,b,c,a,c,d])
        meshes.append(f'<mesh name="{name}" vertex="{" ".join(map(str,vertices))}" face="{" ".join(map(str,faces))}"/>')
        return f'<geom name="{name}" type="mesh" mesh="{name}" pos="0 {center_y} {z}" material="{material}"/>'

    neck = ring('neck_rim', .030, .255, .002, .004, 'plastic')
    spout = ring('spout', .022, .261, .0025, .005, 'seal', slope=.006)
    lid_wall = ring('lid_wall', .033, .008, .0025, .009, 'plastic', center_y=-.031)

    # Inward-facing hook projects from a stem OUTSIDE the slot mouth. Its lower
    # retaining face sits above the slot sill; its upper bevel cams on closure.
    tooth_yz = [(-.0055, .0187), (-.002, .0187), (-.002, .0195), (-.0055, .02065)]
    vertices = [v for x in (-.0025, .0025) for y,z in tooth_yz for v in (x,y,z)]
    faces = [0,2,1,0,3,2,4,5,6,4,6,7]
    for i in range(4):
        j=(i+1)%4
        faces.extend([i,j,j+4,i,j+4,i+4])
    meshes.append(f'<mesh name="hook_prism" vertex="{" ".join(map(str,vertices))}" face="{" ".join(map(str,faces))}"/>')

    catch = []
    # Straight legs and a circular crown follow the photographed U-loop.
    # Its plane is forward of the lid shell and its crown surrounds the separate
    # molded slot roof. Every slot wall participates in mechanism collisions.
    points = [( .015,0,0)] + [(.015*math.cos(i*math.pi/16),0,
                  .0125+.015*math.sin(i*math.pi/16)) for i in range(17)] + [(-.015,0,0)]
    for i, (p, q) in enumerate(zip(points, points[1:])):
        catch.append(f'<geom name="catch_arc_{i}" type="capsule" size=".003" '
                     f'fromto="{" ".join(map(str, p+q))}" material="catch"/>')
    # True hollow cap receiver: floor, side walls, and a decomposed arched roof.
    # Each roof sector is convex; their union preserves the actual empty slot.
    # No convex hull spans the hole, and no support rods cross the hook path.
    receiver = []
    start = math.asin(.0045/.0102)
    for i in range(10):
        a = start+i*(math.pi-2*start)/10
        b = start+(i+1)*(math.pi-2*start)/10
        cross = [(r*math.cos(t), -.0045+r*math.sin(t))
                 for r,t in [(.0117,a),(.0117,b),(.0087,b),(.0087,a)]]
        verts = [v for y in (-.079,-.065) for x,z in cross for v in (x,y,z)]
        roof_faces = [0,2,1,0,3,2,4,5,6,4,6,7]
        for j in range(4):
            k=(j+1)%4
            roof_faces.extend([j,k,k+4,j,k+4,j+4])
        name=f'cap_slot_roof_{i}'
        meshes.append(f'<mesh name="{name}" vertex="{" ".join(map(str,verts))}" face="{" ".join(map(str,roof_faces))}"/>')
        receiver.append(f'<geom name="{name}" type="mesh" mesh="{name}" material="plastic"/>')
    receiver += [
        '<geom name="cap_slot_floor" type="box" pos="0 -.0725 -.0003" size=".0095 .0055 .001" material="plastic"/>',
        '<geom name="cap_slot_left" type="box" pos="-.0085 -.06875 .0005" size=".0015 .00525 .002" material="plastic"/>',
        '<geom name="cap_slot_right" type="box" pos=".0085 -.06875 .0005" size=".0015 .00525 .002" material="plastic"/>',
        '<geom name="cap_slot_back" type="box" pos="0 -.0635 .002" size=".0095 .001 .0035" material="plastic"/>',
    ]
    receiver_names = [f'cap_slot_roof_{i}' for i in range(10)] + ['cap_slot_floor','cap_slot_left','cap_slot_right','cap_slot_back']
    pairs = [f'<pair geom1="{part}" geom2="{button}"/>'
             for part in receiver_names for button in ('rocker_hook','rocker_spine','button_face')]
    pairs += [f'<pair geom1="{part}" geom2="catch_arc_{i}"/>'
              for i in range(len(points)-1) for part in receiver_names]
    xml = f'''<mujoco model="Photo bottle — contact rocker latch and safety loop">
  <compiler angle="radian"/>
  <option timestep="0.00005" integrator="implicitfast" gravity="0 0 -9.81" iterations="100"/>
  <visual><global offwidth="1000" offheight="900"/><quality shadowsize="2048"/>
    <headlight ambient=".35 .35 .35" diffuse=".65 .65 .65" specular=".2 .2 .2"/>
    <rgba haze=".09 .11 .14 1"/></visual>
  <statistic center="0 0 .16" extent=".4"/>
  <default>
    <geom contype="0" conaffinity="0" density="700" solref=".003 1" solimp=".99 .999 .0001"/>
    <joint limited="true" solreflimit=".0008 1"/>
    <pair condim="3" friction=".12 .12 .001 .0001 .0001" solref=".0001 1" solimp=".99 .999 .0001"/>
  </default>
  <asset>
    {''.join(meshes)}
    <texture type="skybox" builtin="gradient" rgb1=".12 .16 .19" rgb2=".07 .10 .13" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" rgb1=".12 .15 .18" rgb2=".14 .17 .20" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="5 5" reflectance=".06"/>
    <material name="steel" rgba=".48 .49 .45 1" specular=".65" shininess=".5"/>
    <material name="plastic" rgba=".15 .17 .16 1" specular=".35" shininess=".2"/>
    <material name="catch" rgba=".26 .29 .27 1" specular=".4"/>
    <material name="seal" rgba=".78 .74 .61 1" specular=".1"/>
    <material name="dark" rgba=".055 .065 .062 1"/>
    <material name="cam" rgba=".96 .62 .20 1"/>
    <material name="tab" rgba=".2 .8 .72 1"/>
  </asset>
  <worldbody>
    <light pos="-.3 -.5 .9" dir=".3 .5 -.9" diffuse=".85 .85 .82" castshadow="true"/>
    <light pos=".4 .2 .5" dir="-.4 -.2 -.4" diffuse=".6 .7 .8" castshadow="false"/>
    <geom name="table" type="plane" size="1 1 .01" material="floor"/>
    <body name="bottle">
      <geom name="base" type="cylinder" size=".035 .006" pos="0 0 .006" material="dark"/>
      <geom name="bottle_wall" type="cylinder" size=".036 .104" pos="0 0 .113" material="steel"/>
      <geom name="shoulder" type="ellipsoid" size=".036 .036 .016" pos="0 0 .215" material="steel"/>
      <geom name="collar" type="cylinder" size=".0355 .017" pos="0 0 .235" material="plastic"/>
      <geom name="mouth_well" type="cylinder" size=".028 .001" pos="0 0 .253" material="dark"/>
      {neck}
      {spout}
      <geom name="hinge_base" type="capsule" size=".005" fromto="-.018 .031 .256 .018 .031 .256" material="plastic"/>
      <body name="lid" pos="0 .031 .257">
        <inertial pos="0 -.030 .010" mass=".025" diaginertia=".000015 .000010 .000014"/>
        <joint name="lid_hinge" axis="-1 0 0" range="0 2.15" damping=".0045" stiffness=".018" springref="2.1"/>
        <geom name="lid_top" type="cylinder" pos="0 -.031 .020" size=".0355 .004" material="plastic"/>
        {lid_wall}
        <geom name="top_badge" type="cylinder" pos="0 -.031 .0242" size=".014 .0004" material="catch"/>
        <geom name="lid_gasket" type="cylinder" pos="0 -.031 .012" size=".024 .002" material="dark"/>
        <geom name="silicone_plug" type="ellipsoid" pos="0 -.031 .005" size=".019 .019 .009" material="seal"/>
        {''.join(receiver)}
      </body>
      <body name="button" pos="0 -.043 .240">
        <inertial pos="0 0 0" mass=".007" diaginertia=".0000005 .0000005 .0000003"/>
        <joint name="button_hinge" axis="1 0 0" range="0 .42" stiffness=".10" damping=".0007"/>
        <geom name="button_face" type="capsule" size=".0065" fromto="0 -.001 -.009 0 -.001 .008" material="catch"/>
        <geom name="rocker_spine" type="box" pos="0 -.0062 .010" size=".0025 .001 .0087" material="plastic"/>
        <geom name="button_press_mark" type="cylinder" pos="0 -.0075 -.012" euler="1.570796 0 0" size=".0045 .0005" material="plastic"/>
        <geom name="rocker_hook" type="mesh" mesh="hook_prism" material="plastic"/>
        <site name="finger_point" pos="0 0 -.012" size=".001" rgba="0 0 0 0"/>
      </body>
      <body name="safety_catch" pos="0 -.0465 .240">
        <inertial pos="0 0 .013" mass=".006" diaginertia=".0000018 .0000022 .0000010"/>
        <joint name="catch_hinge" axis="1 0 0" range="0 3.141593" damping=".0015" frictionloss=".002"/>
        {''.join(catch)}
        <geom name="catch_pin_left" type="cylinder" pos="-.013 0 0" euler="0 1.570796 0" size=".003 .003" material="plastic"/>
        <geom name="catch_pin_right" type="cylinder" pos=".013 0 0" euler="0 1.570796 0" size=".003 .003" material="plastic"/>
      </body>
    </body>
  </worldbody>
  <contact>{''.join(pairs)}</contact>
  <actuator>
    <position name="hand_catch" joint="catch_hinge" kp=".18" kv=".005" ctrlrange="0 3.141593" forcerange="-.06 .06"/>
    <motor name="finger_press" joint="button_hinge" gear=".012" ctrlrange="0 8"/>
    <motor name="hand_close" joint="lid_hinge" gear="-1" ctrlrange="0 .22"/>
  </actuator>
</mujoco>'''
    return xml
