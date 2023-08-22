import pymeshlab as pml

def meshDecimator(vertices, faces, target, remesh = False, optimalPlacement = True):
# def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    OgVertexShape = vertices.shape
    OgFaceShape = faces.shape

    # if backend == 'pyfqmr':
    #     import pyfqmr
    #     solver = pyfqmr.Simplify()
    #     solver.setMesh(verts, faces)
    #     solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
    #     verts, faces, normals = solver.getMesh()
    # else:
        
    mesh = pml.Mesh(vertices, faces)
    meshSet = pml.MeshSet()
    meshSet.add_mesh(mesh, 'mesh') # will copy!

    # filters
    # meshSet.meshing_decimation_clustering(threshold=pml.Percentage(1))
    meshSet.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalPlacement)

    if remesh:
        # meshSet.apply_coord_taubin_smoothing()
        meshSet.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    mesh = meshSet.current_mesh()
    vertices = mesh.vertex_matrix()
    vertices = mesh.face_matrix()

    print(f'[INFO] mesh decimation: {OgVertexShape} --> {vertices.shape}, {OgFaceShape} --> {faces.shape}')

    return vertices, faces


def meshCleaner(vertices, faces, vPct = 1, minF = 8, minD = 5, repair = True, remesh = True, remeshSize = 0.1):
# def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True, remesh_size=0.01):
    # verts: [N, 3]
    # faces: [N, 3]

    OgVertexShape = vertices.shape
    OgFaceShape = faces.shape

    mesh = pml.Mesh(vertices, faces)
    meshSet = pml.MeshSet()
    meshSet.add_mesh(mesh, 'mesh') # will copy!

    # filters
    meshSet.meshing_remove_unreferenced_vertices() # verts not refed by any faces

    if vPct > 0:
        meshSet.meshing_merge_close_vertices(threshold=pml.Percentage(vPct)) # 1/10000 of bounding box diagonal

    meshSet.meshing_remove_duplicate_faces() # faces defined by the same verts
    meshSet.meshing_remove_null_faces() # faces with area == 0

    if minD > 0:
        meshSet.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(minD))
    
    if minF > 0:
        meshSet.meshing_remove_connected_component_by_face_number(mincomponentsize=minF)

    if repair:
        # meshSet.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        meshSet.meshing_repair_non_manifold_edges(method=0)
        meshSet.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    if remesh:
        # meshSet.apply_coord_taubin_smoothing()
        meshSet.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(remeshSize))

    # extract mesh
    mesh = meshSet.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    print(f'[INFO] mesh cleaning: {OgVertexShape} --> {vertices.shape}, {OgFaceShape} --> {faces.shape}')

    return vertices, faces    