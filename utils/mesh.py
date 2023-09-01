import pymeshlab as pml

def meshDecimator(vertices, faces, target, remesh = False, optimalPlacement = True):
    """
    Decimate a 3D mesh while preserving its shape.

    Args:
        vertices (numpy.ndarray): The vertices of the input mesh.
        faces (numpy.ndarray): The faces of the input mesh.
        target (int): The target number of faces after decimation.
        remesh (bool, optional): Whether to remesh the mesh after decimation. Defaults to False.
        optimalPlacement (bool, optional): Whether to use optimal placement during decimatin. Defaults to True.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The vertices and faces of the decimated mesh.
    """

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
    """
    Clean and repair a 3D mesh.

    Args:
        vertices (numpy.ndarray): The vertices of the input mesh.
        faces (numpy.ndarray): The faces of the input mesh.
        vPct (int, optional): Percentage of close vertices of merge. Defaults to 1.
        minF (int, optional): Minimum number of faces in connected components to keep. Defaults to 8.
        minD (int, optional): Minimum diameter of connected components to keep. Defaults to 5.
        repair (bool, optional): Whether to repair non-manifold edges and vertices. Defaults to True.
        remesh (bool, optional): Whether to remesh the mesh after cleaning. Defaults to True.
        remeshSize (float, optional): Target edge length for remeshing. Defaults to 0.01.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The vertices and faces of the cleaned and repaired mesh.
    """

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