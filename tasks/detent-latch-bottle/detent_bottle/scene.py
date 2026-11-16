"""Fixed-layout free bottle with two contact-actuated robot grippers."""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET
from .bottle_model import scene_xml as bottle_scene
from .robot_model import scene as robot_scene

def scene_xml():
    root = ET.fromstring(robot_scene(0)[0])
    root.set('model','dual_arm_free_bottle')
    world = root.find('worldbody')
    ET.SubElement(root.find('visual'),'headlight',ambient='.5 .5 .5',diffuse='.5 .5 .5',specular='.15 .15 .15')
    ET.SubElement(root.find('asset'),'texture',type='skybox',builtin='gradient',rgb1='.22 .27 .33',rgb2='.08 .11 .15',width='256',height='1536')
    for name in ('bar','post','cradle_left','cradle_right'):
        world.remove(world.find(f"body[@name='{name}']"))
    bottle = ET.fromstring(bottle_scene())
    # Preserve the small latch's timestep, exact inertias, joint limits and all
    # 294 contact pairs. Named defaults isolate it from robot armature/friction.
    root.find('option').set('timestep', bottle.find('option').get('timestep'))
    root.find('option').set('cone','elliptic')
    defaults=ET.SubElement(root.find('default'),'default',{'class':'bottle'})
    for element in bottle.find('default'):
        defaults.append(deepcopy(element))
    defaults.find('joint').set('armature','0')
    # External robot/table contact uses bit 1, bottle bit 2. Internal bottle
    # contact still uses its original explicit pair set (no shell self-contact).
    for geom in root.iter('geom'):
        if geom.get('contype') != '0': geom.set('conaffinity','3')
    for item in bottle.find('asset'):
        if item.tag in ('mesh','material') and item.get('name') != 'floor':
            root.find('asset').append(deepcopy(item))
    body=deepcopy(bottle.find("worldbody/body[@name='bottle']"))
    body.set('pos','0 0 .8005');body.set('childclass','bottle')
    ET.SubElement(body,'freejoint',name='bottle_free')
    ET.SubElement(body,'inertial',pos='0 0 .11',mass='.35',diaginertia='.00153 .00153 .000227')
    # Primitive outer shells and convex mechanism pieces are physical; cosmetic
    # rings stay visual to avoid filling their holes with a convex hull.
    solid=('base','bottle_wall','shoulder','collar','lid_top','button_face',
           'rocker_spine','rocker_hook','hinge_base')
    for geom in body.iter('geom'):
        name=geom.get('name','')
        if name in solid or name.startswith(('cap_slot_','catch_arc_')):
            geom.set('contype','2');geom.set('conaffinity','1')
            geom.set('friction','1 .01 .001')
    # The lid wall is a hollow mesh. Convex-hull collision would fill its
    # cavity, so use a ring of small wall pieces for robot contact instead.
    lid=body.find("body[@name='lid']")
    for i in range(32):
        angle=2*math.pi*i/32
        ET.SubElement(lid,'geom',name=f'lid_wall_contact_{i}',type='box',
                      pos=f'{.033*math.cos(angle)} {-.031+.033*math.sin(angle)} .008',
                      euler=f'0 0 {angle}',size='.002 .00325 .009',
                      contype='2',conaffinity='1',mass='0',rgba='0 0 0 0',friction='1 .01 .001')
    world.append(body)
    for pair in bottle.find('contact'):
        pair=deepcopy(pair);pair.set('class','bottle');root.find('contact').append(pair)
    # Bottle actuator subtree is deliberately NOT copied. No welds/equalities.
    for side in ('left','right'):
        hand=world.find(f".//body[@name='{side}_right_hand']")
        hand.find(f"site[@name='{side}_tip']").set('pos','0 0 .142')
        for finger,sign in [('a',-1),('b',1)]:
            name=f'{side}_finger_{finger}'
            g=world.find(f".//geom[@name='{name}_pad']")
            g.set('pos',f'{sign*.004} 0 .130');g.set('size','.004 .004 .012')
            g.set('friction','1.5 .01 .001');g.set('solref','.001 1')
            b=world.find(f".//body[@name='{name}']")
            ET.SubElement(b,'geom',name=name+'_shaft',type='box',pos=f'{sign*.004} 0 .09',
                          size='.004 .005 .028',mass='.025',rgba='.40 .44 .48 1',conaffinity='3')
            touch=b.find(f"site[@name='{name}_touch']")
            touch.set('pos',f'{sign*.001} 0 .130');touch.set('size','.003 .005 .013')
    if True:
        # Rigid on each hand, 70 mm along tool -Y, aimed just past the fingertips.
        for side in ('left','right'):
            ET.SubElement(world.find(f".//body[@name='{side}_right_hand']"),'camera',name=side+'_wrist',
                          pos='0 -.07 .06',xyaxes='-1 0 0 0 .789 -.614',fovy='70')
    # Opposed workspaces leave access to the front button and bottle side.
    world.find("body[@name='left_base']").set('pos','-.40 -.30 .8')
    right=world.find("body[@name='right_base']")
    right.set('pos','.48 .27 .8');right.set('quat','0 0 0 1')
    return ET.tostring(root,encoding='unicode')
