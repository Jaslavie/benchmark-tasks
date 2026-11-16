"""Native separation checks. No force threshold is used."""
import numpy as np
_base=None
_views={}
_state={}
_ready=False
_error=None


def reset():
    global _base,_views,_state,_ready,_error
    _base=None;_views={};_state={};_ready=False;_error=None


def array(v):
    if hasattr(v,'detach'):return v.detach().cpu().numpy()
    return np.asarray(v.numpy() if hasattr(v,'numpy') else v)


def any_close(separations,counts,starts,gap):
    sep=np.asarray(separations).reshape(-1)
    counts=np.asarray(counts).reshape(-1);starts=np.asarray(starts).reshape(-1)
    if len(counts)!=len(starts):raise ValueError('Native contact groups differ')
    for n,s in zip(counts,starts):
        n,s=int(n),int(s)
        if n<0 or s<0 or s+n>len(sep):raise ValueError('Invalid native contact range')
        values=sep[s:s+n]
        if not np.isfinite(values).all():raise ValueError('Nonfinite contact separation')
        if len(values) and values.min()<=gap:return True
    return False


def poll(manager):
    global _state,_ready,_error
    try:
        for name,view in _views.items():
            _,_,_,sep,counts,starts=view.get_contact_data(dt=manager.get_physics_dt())
            _state[name]=any_close(array(sep),array(counts),array(starts),.0001 if name in ('tip','pad') else .001)
        _ready=bool(_views)
    except Exception as exc:_error=repr(exc)


def bind(base):
    global _base,_views
    if _base is base and _views:return
    import carb
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_physx.physics.physx_manager import PhysxManager
    from pxr import UsdPhysics
    carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
    bodies={};robots=[];left=[]
    for prim in get_current_stage().Traverse():
        p=prim.GetPath().pathString
        if '/env_0/' not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        for owner,link,key in [('vise','rotor','rotor'),('vise','pad','pad'),
                               ('indicator','probe','probe'),('indicator','carriage','carriage')]:
            if owner in p.split('/') and p.endswith('/'+link):bodies[key]=p
        if any(a in p for a in ('/LeftArm/','/RightArm/')) and 'PhysxContactReportAPI' in prim.GetAppliedSchemas():
            robots.append(p)
            if '/LeftArm/' in p:left.append(p)
    if len(bodies)!=4 or not robots or not left:raise RuntimeError('Tramming contact bodies missing')
    physics=PhysxManager.get_physics_sim_view()
    def view(sensors,targets):
        return physics.create_rigid_contact_view(sensors,filter_patterns=[targets]*len(sensors),max_contact_data_count=128*len(sensors))
    _views={'tip':view([bodies['probe']],[bodies['rotor']]),
            'pad':view([bodies['pad']],[bodies['rotor']]),
            'robot':view(robots,[bodies['rotor'],bodies['carriage']]),
            'left':view(left,[bodies['rotor']])}
    _base=base
    if not getattr(PhysxManager.step,'_vise_tram_contacts',False):
        original=PhysxManager.step.__func__
        def step(cls):original(cls);poll(cls)
        step._vise_tram_contacts=True
        PhysxManager.step=classmethod(step)


def status():
    if _error:raise RuntimeError('Native tramming contact monitor failed: '+_error)
    return _ready,dict(_state)
