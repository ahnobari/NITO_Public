import numpy as np


def generate_structured_mesh(dim, nel):
    """
    This function generates a structured mesh for a given dimension and number of elements.

    Parameters:
        dim (np.array): Array with the dimension of the domain. Either (2,) or (3,)
        nel (np.array): Array with the number of elements in each direction. Either (2,) or (3,)

    Returns:
        node_positions (np.array): Array with the nodal positions. Shape (n_nodes, dim)
        elements (np.array): Array with the elements. Shape (nel**dim, 4 or 8) for 2D or 3D respectively (Quad and Hex Elements).
    """

    if len(dim) != len(nel):
        raise Exception("Dimensions of dim and nel do not match")

    if len(dim) == 2:

        nx, ny = nel[0] + 1, nel[1] + 1
        num_elem = nel[0] * nel[1]

        L, H = dim[0], dim[1]

        # Create Structured Mesh
        node_positions = np.concatenate(
            np.expand_dims(
                np.meshgrid(np.linspace(0, L, nx), np.linspace(0, H, ny)), -1
            ),
            -1,
        ).reshape(-1, 2)

        elements = np.zeros([num_elem, 4], dtype=int)

        counter = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                elements[counter] = [
                    j * nx + i,
                    j * nx + i + 1,
                    (j + 1) * nx + i + 1,
                    (j + 1) * nx + i,
                ]
                counter += 1
    elif len(dim) == 3:
        nx, ny, nz = nel[0] + 1, nel[1] + 1, nel[2] + 1
        num_elem = nel[0] * nel[1] * nel[2]

        L, H, W = dim[0], dim[1], dim[2]

        # Create Structured Mesh
        node_positions = np.concatenate(
            np.expand_dims(
                np.meshgrid(
                    np.linspace(0, L, nx), np.linspace(0, H, ny), np.linspace(0, W, nz)
                ),
                -1,
            ),
            -1,
        ).reshape(-1, 3)

        elements = np.zeros([num_elem, 8], dtype=int)

        counter = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # [k*nx*ny+j*nx+i,k*nx*ny+j*nx+i+1,k*nx*ny+(j+1)*nx+i+1,k*nx*ny+(j+1)*nx+i,
                    #                     (k+1)*nx*ny+j*nx+i,(k+1)*nx*ny+j*nx+i+1,(k+1)*nx*ny+(j+1)*nx+i+1,(k+1)*nx*ny+(j+1)*nx+i]
                    elements[counter] = [
                        nz * i + nz * nx * j + k,
                        nz * (i + 1) + nz * nx * j + k,
                        nz * (i + 1) + nz * nx * (j + 1) + k,
                        nz * i + nz * nx * (j + 1) + k,
                        nz * i + nz * nx * j + k + 1,
                        nz * (i + 1) + nz * nx * j + k + 1,
                        nz * (i + 1) + nz * nx * (j + 1) + k + 1,
                        nz * i + nz * nx * (j + 1) + k + 1,
                    ]
                    counter += 1
    else:
        raise Exception("Only 2D and 3D meshes are supported")

    return elements, node_positions


triangle_args = ["-setnumber", "Mesh.SaveWithoutOrphans", "1"]
quad_args = [
    "-setnumber",
    "Mesh.Algorithm",
    "6",
    "-setnumber",
    "Mesh.RecombinationAlgorithm",
    "3",
    "-setnumber",
    "Mesh.RecombineAll",
    "1",
    "-setnumber",
    "Mesh.SaveWithoutOrphans",
    "1",
]
tet_args = ["-setnumber", "Mesh.SaveWithoutOrphans", "1"]
hex_args = [
    "-setnumber",
    "Mesh.Algorithm",
    "8",
    "-setnumber",
    "Mesh.RecombinationAlgorithm",
    "3",
    "-setnumber",
    "Mesh.RecombineAll",
    "1",
    "-setnumber",
    "Mesh.SubdivisionAlgorithm",
    "2",
    "-setnumber",
    "Mesh.Algorithm3D",
    "1",
    "-setnumber",
    "Mesh.Recombine3DAll",
    "1",
    "-setnumber",
    "Mesh.SaveWithoutOrphans",
    "1",
    "-setnumber",
    "Mesh.Smoothing",
    "5",
]

tri_quad_args = [
    "-setnumber",
    "Mesh.Algorithm",
    "6",
    "-setnumber",
    "Mesh.RecombinationAlgorithm",
    "5",
    "-setnumber",
    "Mesh.RecombineAll",
    "1",
    "-setnumber",
    "Mesh.SaveWithoutOrphans",
    "1",
]

tet_hex_args = [
    "-setnumber",
    "Mesh.Algorithm",
    "8",
    "-setnumber",
    "Mesh.RecombinationAlgorithm",
    "5",
    "-setnumber",
    "Mesh.RecombineAll",
    "1",
    "-setnumber",
    "Mesh.SubdivisionAlgorithm",
    "2",
    "-setnumber",
    "Mesh.Algorithm3D",
    "1",
    "-setnumber",
    "Mesh.Recombine3DAll",
    "1",
    "-setnumber",
    "Mesh.SaveWithoutOrphans",
    "1",
    "-setnumber",
    "Mesh.Smoothing",
    "5",
]
