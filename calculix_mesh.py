"""Generate a provisional free-free CalculiX S8R shell modal deck.

The separated hub plates are represented by one coplanar equivalent shell
whose areal mass, membrane stiffness, and bending stiffness match the idealized
pair. Four arm surfaces connect through clamp interpolation equations.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from geometry import layout, load_design


def _patches(design: dict) -> list[dict]:
    h = design["hub_width_m"] / 2
    tp = design["hub_plate_thickness_m"]
    gap = design["geometry"]["hub_plate_gap_m"]
    square = [(-h,-h), (h,-h), (h,h), (-h,h)]
    centre_offset = (gap + tp) / 2
    equivalent_thickness = math.sqrt(tp**2 + 12 * centre_offset**2)
    equivalent_scale = 2 * tp / equivalent_thickness
    patches = [{"name": "HUB_EQUIV", "z": 0.0, "corners": square,
                "thickness": equivalent_thickness, "material_scale": equivalent_scale,
                "axis": (1.0,0.0,0.0)}]
    from geometry import arm_polygon
    for arm in layout(design)["arms"]:
        # Reorder polygon so bilinear patch axes follow arm length and width.
        p = arm_polygon(design, arm)
        patches.append({"name": arm["name"].upper(), "z": 0.0,
                        "corners": p, "thickness": design["arm_thickness_m"],
                        "material_scale": 1.0,
                        "axis": (*arm["direction"],0.0)})
    return patches


def _dimensions(patch: dict, mesh_size_m: float,
                width_division_scale: float = 1.0) -> tuple[int, int]:
    p = patch["corners"]
    along = math.dist(p[0],p[1]); across = math.dist(p[1],p[2])
    # Even counts place nodes at the hub centre and motor centre.
    return (max(2, 2*math.ceil(along/(2*mesh_size_m))),
            max(2, 2*math.ceil(across*width_division_scale/(2*mesh_size_m))))


def _interp(p: list[tuple[float,float]], s: float, t: float, z: float) -> tuple[float,float,float]:
    weights = ((1-s)*(1-t), s*(1-t), s*t, (1-s)*t)
    return (sum(w*q[0] for w,q in zip(weights,p)),
            sum(w*q[1] for w,q in zip(weights,p)), z)


def _structured_mesh(patches: list[dict], mesh_size_m: float,
                     width_division_scale: float = 1.0):
    nodes: dict[int,tuple[float,float,float]] = {}
    elements: dict[int,tuple[int,list[int]]] = {}
    lookup: dict[tuple[int,tuple[float,float,float]],int] = {}
    def node(pid, xyz):
        key = (pid, tuple(round(v, 12) for v in xyz))
        if key not in lookup:
            idx = len(nodes)+1; lookup[key] = idx; nodes[idx] = xyz
        return lookup[key]
    for pid, patch in enumerate(patches, start=1):
        ni,nj = _dimensions(patch,mesh_size_m,width_division_scale)
        p,z = patch["corners"],patch["z"]
        for i in range(ni):
            for j in range(nj):
                s0,s1=i/ni,(i+1)/ni; t0,t1=j/nj,(j+1)/nj
                coords=((s0,t0),(s1,t0),(s1,t1),(s0,t1),
                        ((s0+s1)/2,t0),(s1,(t0+t1)/2),
                        ((s0+s1)/2,t1),(s0,(t0+t1)/2))
                conn=[node(pid, _interp(p,s,t,z)) for s,t in coords]
                elements[len(elements)+1]=(pid,conn)
    return nodes,elements


def _geo_text(patches: list[dict], mesh_size_m: float) -> str:
    lines=[f"Mesh.CharacteristicLengthMin = {mesh_size_m:.12g};",
           f"Mesh.CharacteristicLengthMax = {mesh_size_m:.12g};",
           "Mesh.ElementOrder = 2;", "Mesh.SecondOrderIncomplete = 1;",
           "Mesh.MshFileVersion = 2.2;"]
    point_id=line_id=1
    for pid,patch in enumerate(patches,start=1):
        corner_ids=[]; line_ids=[]
        for x,y in patch["corners"]:
            lines.append(f"Point({point_id}) = {{{x:.12g},{y:.12g},{patch['z']:.12g},{mesh_size_m:.12g}}};")
            corner_ids.append(point_id); point_id+=1
        ni,nj=_dimensions(patch,mesh_size_m)
        for k in range(4):
            a,b=corner_ids[k],corner_ids[(k+1)%4]
            lines.append(f"Line({line_id}) = {{{a},{b}}};")
            lines.append(f"Transfinite Curve {{{line_id}}} = {(ni if k%2==0 else nj)+1};")
            line_ids.append(line_id); line_id+=1
        joined=','.join(map(str,line_ids))
        lines.extend((f"Curve Loop({pid}) = {{{joined}}};",
                      f"Plane Surface({pid}) = {{{pid}}};",
                      f"Transfinite Surface {{{pid}}};",
                      f"Recombine Surface {{{pid}}};",
                      f"Physical Surface({pid}) = {{{pid}}};"))
    return '\n'.join(lines)+'\n'


def _parse_msh2(path: Path):
    lines=path.read_text().splitlines()
    try:
        ns=lines.index('$Nodes'); es=lines.index('$Elements')
    except ValueError as exc:
        raise ValueError('Gmsh did not produce MSH2 Nodes/Elements sections') from exc
    n=int(lines[ns+1]); nodes={}
    for row in lines[ns+2:ns+2+n]:
        cols=row.split(); nodes[int(cols[0])]=tuple(map(float,cols[1:4]))
    count=int(lines[es+1]); elements={}
    for row in lines[es+2:es+2+count]:
        cols=list(map(int,row.split())); element_id,etype,ntags=cols[:3]
        if etype==16:
            if ntags<1: raise ValueError('Gmsh quad missing physical surface tag')
            elements[element_id]=(cols[3],cols[3+ntags:])
    if not elements or any(len(c)!=8 for _,c in elements.values()):
        raise ValueError('Gmsh must output incomplete quadratic quads (type 16) only')
    return nodes,elements


def _closest(nodes: dict, point: tuple[float,float,float], allowed: set[int] | None=None) -> int:
    ids=allowed if allowed is not None else nodes.keys()
    return min(ids,key=lambda i: math.dist(nodes[i],point))


def _write_deck(path: Path, design: dict, patches: list[dict], nodes: dict,
                elements: dict, mode_count: int,
                hub_stiffness_multiplier: float = 1.0) -> tuple[list[int],list[list[float]],float,dict]:
    # Keep patch node sets separate; Gmsh and fallback use identical patch tags.
    patch_nodes={pid:set() for pid in range(1,len(patches)+1)}
    patch_elems={pid:[] for pid in range(1,len(patches)+1)}
    for eid,(pid,conn) in elements.items():
        patch_nodes[pid].update(conn); patch_elems[pid].append(eid)
    if any(not patch_elems[p] for p in patch_elems):
        raise ValueError('Each hub/arm surface needs at least one S8R element')
    loc=layout(design)
    tip_positions=[[*arm['motor_xy_m'],0.0] for arm in loc['arms']]
    tip_ids=[_closest(nodes,tuple(xyz),patch_nodes[pid]) for pid,xyz in enumerate(tip_positions,start=2)]
    for i,tip in enumerate(tip_ids):
        if math.dist(nodes[tip],tip_positions[i])>1e-8:
            raise ValueError('Motor centre is not a shell mesh node')
    hub_center=_closest(nodes,(0,0,0),patch_nodes[1])
    if math.hypot(*nodes[hub_center][:2])>1e-8:
        raise ValueError('Hub centre is not a shell mesh node')
    g=design['geometry']; rho=design['rho_kg_m3']
    mount_mass=rho*math.pi*(g['motor_mount_diameter_m']/2)**2*g['motor_mount_thickness_m']
    mass_estimate=(2*rho*design['hub_width_m']**2*design['hub_plate_thickness_m']
                   +4*rho*design['arm_width_m']*design['arm_thickness_m']
                   *(design['arm_length_m']+g['arm_clamp_overlap_m'])
                   +4*(design['motor_mass_kg']+mount_mass)+design['payload_mass_kg'])
    lines=['*HEADING','X quad provisional free-free S8R shell model; SI units',
           '** Materials are illustrative equivalent orthotropic constants; replace with supplier/measurement data.',
           '*NODE']
    for nid,xyz in sorted(nodes.items()):
        lines.append(f"{nid}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")
    for pid,patch in enumerate(patches,start=1):
        lines.append(f"*ELEMENT, TYPE=S8R, ELSET={patch['name']}")
        for eid in patch_elems[pid]:
            lines.append(', '.join(map(str,[eid]+elements[eid][1])))
    E1,E2,E3=design['E_long_Pa'],design['E_trans_Pa'],design['E_through_Pa']
    G12,G13,G23=design['G_Pa'],design['G13_Pa'],design['G23_Pa']
    for material,scale in (("CARBON_EQUIV",1.0),("HUB_EQUIV",patches[0]['material_scale'])):
        elastic_scale=scale*(hub_stiffness_multiplier if material=="HUB_EQUIV" else 1.0)
        lines += [f'*MATERIAL, NAME={material}', '*ELASTIC, TYPE=ENGINEERING CONSTANTS']
        lines.append(f"{E1*elastic_scale:.12g}, {E2*elastic_scale:.12g}, {E3*elastic_scale:.12g}, {design['nu12']:.12g}, {design['nu13']:.12g}, {design['nu23']:.12g}, {G12*elastic_scale:.12g}, {G13*elastic_scale:.12g}")
        lines += [f'{G23*elastic_scale:.12g}', '*DENSITY', f'{rho*scale:.12g}']
    for pid,patch in enumerate(patches,start=1):
        ax=patch['axis']; ay=(-ax[1],ax[0],0.0)
        if pid==1: ay=(0.0,1.0,0.0)
        name=f'ORIENT_{pid}'
        material='HUB_EQUIV' if pid==1 else 'CARBON_EQUIV'
        lines += [f'*ORIENTATION, NAME={name}, SYSTEM=RECTANGULAR',
                  ', '.join(f'{v:.12g}' for v in (*ax,*ay)),
                  f"*SHELL SECTION, ELSET={patch['name']}, MATERIAL={material}, ORIENTATION={name}",
                  f"{patch['thickness']:.12g}"]
    # Each arm node in the clamp footprint follows the bilinear displacement
    # of the four hub corners. Bilinear interpolation exactly reproduces any
    # rigid translation or rotation on the coplanar reference surface.
    h=design['hub_width_m']/2
    hub_corners=[_closest(nodes,(x,y,0.0),patch_nodes[1]) for x,y in patches[0]['corners']]
    if any(math.dist(nodes[nid],(*xy,0.0))>1e-8 for nid,xy in zip(hub_corners,patches[0]['corners'])):
        raise ValueError('Hub corner nodes missing from shell mesh')
    lines += ['** Idealized rigid clamp footprint via hub-corner interpolation.', '*EQUATION']
    equation_count=0
    clamped_nodes=0
    for pid in range(2,6):
        for arm_node in sorted(patch_nodes[pid]):
            x,y,_=nodes[arm_node]
            if max(abs(x),abs(y))>h+1e-10:
                continue
            s=(x+h)/(2*h); t=(y+h)/(2*h)
            weights=((1-s)*(1-t),s*(1-t),s*t,(1-s)*t)
            for dof in (1,2,3):
                terms=[(arm_node,dof,1.0)] + [(nid,dof,-w) for nid,w in zip(hub_corners,weights) if w>1e-12]
                lines.append(str(len(terms)))
                lines.append(', '.join(f'{nid}, {direction}, {coefficient:.12g}' for nid,direction,coefficient in terms))
                equation_count+=1
            clamped_nodes+=1
    if clamped_nodes==0:
        raise ValueError('No arm nodes lie in the hub clamp footprint; refine mesh')
    next_id=max(elements)+1
    for i,nid in enumerate(tip_ids,start=1):
        name=f'MOTOR_MASS_{i}'
        lines += [f'*ELEMENT, TYPE=MASS, ELSET={name}',f'{next_id}, {nid}',
                  f'*MASS, ELSET={name}',f"{design['motor_mass_kg']+mount_mass:.12g}"]
        next_id+=1
    lines += ['*ELEMENT, TYPE=MASS, ELSET=PAYLOAD_MASS',f'{next_id}, {hub_center}',
              '*MASS, ELSET=PAYLOAD_MASS',f"{design['payload_mass_kg']:.12g}"]
    # OUTPUT=2D returns the original shell node IDs; the default expanded 3D
    # output reports generated brick node IDs and cannot be matched to tips.
    lines += ['*STEP','*FREQUENCY',str(mode_count),'*NODE FILE, OUTPUT=2D','U','*END STEP']
    path.write_text('\n'.join(lines)+'\n')
    return tip_ids,tip_positions,mass_estimate,{'node_count':len(nodes),'shell_element_count':len(elements),
                                                'equation_count':equation_count,'clamped_nodes':clamped_nodes,
                                                'mass_element_count':5,'element_type':'S8R',
                                                'hub_model':'coplanar_equivalent_two_plate_shell',
                                                'hub_equivalent_thickness_m':patches[0]['thickness'],
                                                'hub_material_scale':patches[0]['material_scale'],
                                                'hub_stiffness_multiplier':hub_stiffness_multiplier,
                                                'clamp_model':'hub_corner_interpolation_EQUATION'}


def write_calculix_deck(design: dict, out_dir: str | Path,
                        mesh_size_m: float = 0.012, mode_count: int = 18,
                        mesh_generator: str = "auto",
                        width_division_scale: float = 1.0,
                        hub_stiffness_multiplier: float = 1.0) -> dict:
    """Write .inp and mesh manifest; no CalculiX solve is performed.

    Returns absolute ``inp_path``, four ``tip_node_ids`` in X-arm angle order,
    ``tip_locations_m``, ``mass_estimate_kg``, and ``mesh`` metadata.
    """
    if not math.isfinite(mesh_size_m) or mesh_size_m<=0:
        raise ValueError('mesh_size_m must be finite and positive')
    if not isinstance(mode_count,int) or mode_count<7:
        raise ValueError('mode_count must be at least seven (six rigid modes)')
    if mesh_generator not in {"auto", "gmsh", "structured"}:
        raise ValueError("mesh_generator must be auto, gmsh, or structured")
    if not math.isfinite(width_division_scale) or width_division_scale <= 0:
        raise ValueError("width_division_scale must be positive and finite")
    if not math.isfinite(hub_stiffness_multiplier) or hub_stiffness_multiplier <= 0:
        raise ValueError("hub_stiffness_multiplier must be positive and finite")
    # Validate the shared configuration without changing user parameters.
    from geometry import load_design as _load
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'design.json'; p.write_text(json.dumps(design)); _load(p)
    out=Path(out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    patches=_patches(design)
    geo=out/'x_quad_shells.geo'; geo.write_text(_geo_text(patches,mesh_size_m))
    gmsh=shutil.which('gmsh')
    if mesh_generator == "gmsh" and not gmsh:
        raise FileNotFoundError("Gmsh requested but not installed")
    if gmsh and mesh_generator != "structured":
        msh=out/'x_quad_shells.msh'
        run=subprocess.run([gmsh,'-2','-format','msh2','-o',str(msh),str(geo)],
                           capture_output=True,text=True,timeout=120)
        if run.returncode!=0:
            raise RuntimeError(f'Gmsh meshing failed: {run.stderr[-2000:]}')
        nodes,elements=_parse_msh2(msh); generator='native_gmsh_msh2'
    else:
        nodes,elements=_structured_mesh(patches,mesh_size_m,width_division_scale)
        generator='deterministic_structured_fallback'
    deck=out/'x_quad_modal.inp'
    tips,locations,mass,metadata=_write_deck(deck,design,patches,nodes,elements,
                                             mode_count,hub_stiffness_multiplier)
    metadata.update({'generator':generator,'target_size_m':mesh_size_m,
                     'width_division_scale':width_division_scale,
                     'geometry_file':str(geo),'mode_count_requested':mode_count})
    result={'inp_path':str(deck),'tip_node_ids':tips,'tip_locations_m':locations,
            'mass_estimate_kg':mass,'mesh':metadata,
            'provenance':'provisional_shell_deck_unverified_until_ccx_and_physical_test'}
    (out/'calculix_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'config/reference_design.json')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'build/calculix')
    parser.add_argument('--mesh-size-m',type=float,default=0.012)
    parser.add_argument('--mode-count',type=int,default=18)
    args=parser.parse_args()
    print(json.dumps(write_calculix_deck(load_design(args.config),args.out,args.mesh_size_m,args.mode_count),indent=2))
