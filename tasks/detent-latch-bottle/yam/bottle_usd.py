"""Translate the native procedural bottle's links and convex pieces into USD.

Preserves the hollow receiver (ten convex roof sectors), hook wedge, U-loop
capsules and explicit sibling contact whitelist. Cosmetic hollow rings have no
convex hull collider. Geometry is not replaced with software latch conditions.
"""
import math
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from detent_bottle.bottle_model import scene_xml


def nums(text,default=()):
    return tuple(float(x) for x in text.split()) if text else default


def spawn_bottle(prim_path,translation=None,orientation=None):
    import isaaclab.sim as sim
    from isaaclab.sim.utils import bind_physics_material,bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf,PhysxSchema,Sdf,UsdGeom,UsdPhysics
    from yam_arena.station import linear_rgb

    xml=ET.fromstring(scene_xml());stage=get_current_stage()
    root=UsdGeom.Xform.Define(stage,prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0,0,0))))
    x,y,z,w=orientation if orientation is not None else (0,0,0,1)
    root.AddOrientOp().Set(Gf.Quatf(w,x,y,z))
    looks={}
    for mat in xml.findall("asset/material"):
        name=mat.get("name");rgba=nums(mat.get("rgba"),(.5,.5,.5,1))
        path=f"{prim_path}/Looks/{name}"
        cfg=sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(rgba[:3]),roughness=.5)
        cfg.func(path,cfg);looks[name]=path
    material=f"{prim_path}/Looks/contact"
    cfg=sim.RigidBodyMaterialCfg(static_friction=1.0,dynamic_friction=.8)
    cfg.func(material,cfg)
    internal_material=f"{prim_path}/Looks/latch_contact"
    cfg=sim.RigidBodyMaterialCfg(static_friction=.12,dynamic_friction=.12)
    cfg.func(internal_material,cfg)
    meshes={m.get("name"):m for m in xml.findall("asset/mesh")}
    solids={"base","bottle_wall","shoulder","collar","lid_top","button_face",
            "rocker_spine","rocker_hook","hinge_base"}
    geoms={}; owners={}

    def geom(body_path,g):
        name=g.get("name");kind=g.get("type");s=nums(g.get("size"))
        path=f"{body_path}/{name}"
        if kind=="box":
            shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(1)
            scale=tuple(2*v for v in s)
        elif kind=="cylinder":
            shape=UsdGeom.Cylinder.Define(stage,path);shape.CreateRadiusAttr(s[0]);shape.CreateHeightAttr(2*s[1]);scale=None
        elif kind=="ellipsoid":
            shape=UsdGeom.Sphere.Define(stage,path);shape.CreateRadiusAttr(1);scale=s
        elif kind=="capsule":
            ends=np.array(nums(g.get("fromto"))).reshape(2,3)
            delta=ends[1]-ends[0];length=float(np.linalg.norm(delta))
            shape=UsdGeom.Capsule.Define(stage,path);shape.CreateRadiusAttr(s[0]);shape.CreateHeightAttr(length)
            scale=None
        elif kind=="mesh":
            m=meshes[g.get("mesh")]
            vertices=np.array(nums(m.get("vertex"))).reshape(-1,3)
            indices=[int(v) for v in m.get("face").split()]
            shape=UsdGeom.Mesh.Define(stage,path)
            shape.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
            shape.CreateFaceVertexCountsAttr([3]*(len(indices)//3))
            shape.CreateFaceVertexIndicesAttr(indices)
            shape.CreateSubdivisionSchemeAttr("none");scale=None
        else: raise ValueError(kind)
        position=nums(g.get("pos"),(0,0,0))
        rotation=Gf.Quatf(1,0,0,0)
        if kind=="capsule":
            position=tuple(ends.mean(0))
            r=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*(delta/length))).GetQuat()
            rotation=Gf.Quatf(r)
        elif g.get("euler"):
            # MuJoCo default intrinsic xyz; all source cylinders use one axis.
            rx,ry,rz=nums(g.get("euler"))
            r=Gf.Rotation(Gf.Vec3d(1,0,0),math.degrees(rx))*Gf.Rotation(Gf.Vec3d(0,1,0),math.degrees(ry))*Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(rz))
            rotation=Gf.Quatf(r.GetQuat())
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        shape.AddOrientOp().Set(rotation)
        if scale: shape.AddScaleOp().Set(Gf.Vec3f(*scale))
        prim=shape.GetPrim()
        if g.get("material"): bind_visual_material(path,looks[g.get("material")],stage=stage)
        collides=name in solids or name.startswith(("cap_slot_","catch_arc_","lid_wall_contact_"))
        if collides:
            UsdPhysics.CollisionAPI.Apply(prim)
            if kind=="mesh": UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            cp=PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(.0001);cp.CreateRestOffsetAttr(0.)
            internal=name.startswith(("cap_slot_","catch_arc_")) or name in ("button_face","rocker_spine","rocker_hook")
            bind_physics_material(path,internal_material if internal else material,stage=stage)
            geoms[name]=path;owners[name]=body_path

    source=xml.find("worldbody/body[@name='bottle']")
    lid=source.find("body[@name='lid']")
    for i in range(32):
        a=2*math.pi*i/32
        ET.SubElement(lid,"geom",name=f"lid_wall_contact_{i}",type="box",
            pos=f"{.033*math.cos(a)} {-.031+.033*math.sin(a)} .008",
            euler=f"0 0 {a}",size=".002 .00325 .009",material="plastic")
    sources=[source,*source.findall("body")]
    body_paths={b.get("name"):f"{prim_path}/{b.get('name')}" for b in sources}
    for b in sources:
        path=body_paths[b.get("name")];xf=UsdGeom.Xform.Define(stage,path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*nums(b.get("pos"),(0,0,0))))
        xf.AddOrientOp().Set(Gf.Quatf(1,0,0,0))
        prim=xf.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
        inertial=b.find("inertial");mass=UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(.35 if inertial is None else float(inertial.get("mass")))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*((0,0,.11) if inertial is None else nums(inertial.get("pos"),(0,0,0)))))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*((.00153,.00153,.000227) if inertial is None else nums(inertial.get("diaginertia")))))
        props=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        props.CreateMaxDepenetrationVelocityAttr(.5)
        props.CreateSolverPositionIterationCountAttr(64)
        props.CreateSolverVelocityIterationCountAttr(8)
        props.CreateLinearDampingAttr(0.0);props.CreateAngularDampingAttr(0.0)
        for g in b.findall("geom"): geom(path,g)
        j=b.find("joint")
        if j is not None:
            joint=UsdPhysics.RevoluteJoint.Define(stage,f"{prim_path}/joints/{j.get('name')}")
            joint.CreateAxisAttr("X")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_paths["bottle"])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*nums(b.get("pos"))))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
            axis=nums(j.get("axis"))
            rot=Gf.Quatf(0,0,0,1) if axis[0]<0 else Gf.Quatf(1,0,0,0)
            joint.CreateLocalRot0Attr(rot);joint.CreateLocalRot1Attr(rot)
            lo,hi=nums(j.get("range"))
            joint.CreateLowerLimitAttr(math.degrees(lo));joint.CreateUpperLimitAttr(math.degrees(hi))
            joint.CreateCollisionEnabledAttr(False)
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),"angular")
            drive.CreateTypeAttr("force")
            # USD angular-drive stiffness/damping use degrees, Isaac Lab writes
            # the final configured radian gains after articulation initialization.
            drive.CreateStiffnessAttr(float(j.get("stiffness",0))*math.pi/180)
            drive.CreateDampingAttr(float(j.get("damping",0))*math.pi/180)
            drive.CreateTargetPositionAttr(math.degrees(float(j.get("springref",0))))

    # Native explicit mechanical contacts, retaining gaps in the hollow receiver.
    allowed={frozenset((p.get("geom1"),p.get("geom2"))) for p in xml.findall("contact/pair")}
    for n,path in geoms.items():
        excluded=[Sdf.Path(other) for m,other in geoms.items() if owners[m]!=owners[n]
                  and frozenset((n,m)) not in allowed]
        if excluded: UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().SetTargets(excluded)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(body_paths["bottle"]))
    art=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(body_paths["bottle"]))
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSolverPositionIterationCountAttr(64);art.CreateSolverVelocityIterationCountAttr(8)
    report=os.environ.get("BOTTLE_MODEL_REPORT")
    if report:
        Path(report).write_text(json.dumps({
            "rigid_links":list(body_paths),"free_base":True,
            "source_contact_pairs":len(allowed),"colliders":len(geoms),
            "self_collision_enabled":True,"contact_offset_m":.0001,
            "joints":[{"name":b.find("joint").get("name"),
                "axis":nums(b.find("joint").get("axis")),
                "limits_radians":nums(b.find("joint").get("range"))}
                for b in sources if b.find("joint") is not None],
            "direct_object_policy_controls":False},indent=2))
    return root.GetPrim()
