import pymeshlab as pml

def meshDecimator(vertices, faces, target, remesh = False, optimalPlacement = True):

    startVerticesShape = vertices.shape
    startFaceShape = faces.shape

    mesh = pml.Mesh(vertices, faces)
    meshSet = pml.MeshSet()
    meshSet.add_mesh(mesh, 'mesh')
    meshSet.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalPlacement)

    if remesh:
        meshSet.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    mesh = meshSet.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    print(f"Decimating Mesh: {startVerticesShape} ==> {vertices.shape}, {startFaceShape} ==> {faces.shape}")

    return vertices, faces

def meshCleaner(vertices, faces, vPct = 1, minF = 8, minD = 5, repair = True, remesh = True, remeshSize = 0.01):

    startVerticesShape = vertices.shape
    startFaceShape = faces.shape

    mesh = pml.Mesh(vertices, faces)
    meshSet = pml.MeshSet()
    meshSet.add_mesh(mesh, 'mesh')

    meshSet.meshing_remove_unreferenced_vertices()

    if vPct > 0:
        meshSet.meshing_merge_close_vertices(threshold=pml.Percentage(vPct))

    meshSet.meshing_remove_duplicate_faces()
    meshSet.meshing_remove_null_faces()

    if minD > 0:
        meshSet.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(minD))
    
    if minF > 0:
        meshSet.meshing_remove_connected_component_by_face_number(mincomponentsize=minF)

    if repair:
        meshSet.meshing_repair_non_manifold_edges(method=0)
        meshSet.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    if remesh:
        meshSet.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(remeshSize))

    mesh = meshSet.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    print(f"Cleaning Mesh: {startVerticesShape} ==> {vertices.shape}, {startFaceShape} ==> {faces.shape}")

    return vertices, faces    