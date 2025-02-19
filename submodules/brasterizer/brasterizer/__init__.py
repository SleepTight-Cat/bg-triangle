from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ipdb
import numpy as np
from pytorch3d.renderer.utils import parse_image_size
import torch
import torch.nn as nn
import time


from .clip import (
    clip_faces,
    ClipFrustum,
    convert_clipped_rasterization_to_original_faces,
)
from . import _C


def RasterizeMeshesCoarseCuda(face_verts, mesh_to_face_first_idx, num_faces_per_mesh, image_size,blur_radius, bin_size,max_faces_per_bin):
    return _C._rasterize_meshes_coarse(face_verts, mesh_to_face_first_idx, num_faces_per_mesh, image_size,blur_radius, bin_size,max_faces_per_bin)

def RasterizeMeshesFineCuda(
    face_verts,
    bin_faces,
    elems_per_bin,
    clipped_faces_neighbor_idx,
    image_size,
    blur_radius,
    bin_size,
    faces_per_pixel,
    perspective_correct,
    clip_barycentric_coords,
    cull_backfaces
):
    return _C._rasterize_meshes_fine(
        face_verts,
        bin_faces,
        elems_per_bin,
        clipped_faces_neighbor_idx,
        image_size,
        blur_radius,
        bin_size,
        faces_per_pixel,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces
    )

def DetectOutlineVerticesCuda(face_verts, bprimitive_image):
    return _C._detect_outline_vertices(face_verts, bprimitive_image)


def DetectOutlinePixelsCuda( bprimitive_image):
    return _C._detect_outline_pixels( bprimitive_image)


class _RasterizeBMesh(torch.autograd.Function):
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """

    @staticmethod
    def forward(ctx, bpMesh, camera, raster_settings):
        raise NotImplementedError("Forward not implemented for MeshRasterizer")

    @staticmethod
    def backward(ctx, grad_pix_to_face, grad_zbuf, grad_bary_coords, grad_dists, _):
        raise NotImplementedError("Backward not implemented for MeshRasterizer")


@dataclass
class BMeshRasterizationSettings:
    """
    Class to store the mesh rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_faces_opengl: Max number of faces in any mesh we will rasterize. Used only by
            MeshRasterizerOpenGL to pre-allocate OpenGL memory.
        max_faces_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of faces
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_faces_per_bin=None attempts to set with a heuristic.
        perspective_correct: Whether to apply perspective correction when
            computing barycentric coordinates for pixels.
            None (default) means make correction if the camera uses perspective.
        clip_barycentric_coords: Whether, after any perspective correction
            is applied but before the depth is calculated (e.g. for
            z clipping), to "correct" a location outside the face (i.e. with
            a negative barycentric coordinate) to a position on the edge of the
            face. None (default) means clip if blur_radius > 0, which is a condition
            under which such outside-face-points are likely.
        cull_backfaces: Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: Whether to cull triangles outside the view frustum.
            Culling involves removing all faces which fall outside view frustum.
            Default is False for performance as often not needed.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_faces_opengl: int = 10_000_000
    max_faces_per_bin: Optional[int] = None
    perspective_correct: Optional[bool] = None
    clip_barycentric_coords: Optional[bool] = None
    cull_backfaces: bool = False
    z_clip_value: Optional[float] = None
    cull_to_frustum: bool = False


# TODO make the epsilon user configurable
kEpsilon = 1e-8

# Maximum number of faces per bins for
# coarse-to-fine rasterization
kMaxFacesPerBin = 5000


class BMeshRasterizer(nn.Module):
    def __init__(self, backend="cuda") -> None:
        super().__init__()

        self.backend = backend



    def forward(self,
        meshes,
        image_size: Union[int, List[int], Tuple[int, int]] = 256,
        blur_radius: float = 0.0,
        faces_per_pixel: int = 8,
        bin_size: Optional[int] = None,
        max_faces_per_bin: Optional[int] = None,
        perspective_correct: bool = True,
        clip_barycentric_coords: bool = False,
        cull_backfaces: bool = False,
        z_clip_value: Optional[float] = None,
        cull_to_frustum: bool = True,
    ):
        """
        Rasterize a batch of meshes given the shape of the desired output image.
        Each mesh is rasterized onto a separate image of shape
        (H, W) if `image_size` is a tuple or (image_size, image_size) if it
        is an int.

        If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
        the aspect ratio needs special consideration. There are two aspect ratios
        to be aware of:
            - the aspect ratio of each pixel
            - the aspect ratio of the output image
        The camera can be used to set the pixel aspect ratio. In the rasterizer,
        we assume square pixels, but variable image aspect ratio (i.e rectangle images).

        In most cases you will want to set the camera aspect ratio to
        1.0 (i.e. square pixels) and only vary the
        `image_size` (i.e. the output image dimensions in pixels).

        Args:
            meshes: A Meshes object representing a batch of meshes, batch size N.
            image_size: Size in pixels of the output image to be rasterized.
                Can optionally be a tuple of (H, W) in the case of non square images.
            blur_radius: Float distance in the range [0, 2] used to expand the face
                bounding boxes for rasterization. Setting blur radius
                results in blurred edges around the shape instead of a
                hard boundary. Set to 0 for no blur.
            faces_per_pixel (Optional): Number of faces to save per pixel, returning
                the nearest faces_per_pixel points along the z-axis.
            bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
                bin_size=0 uses naive rasterization; setting bin_size=None attempts to
                set it heuristically based on the shape of the input. This should not
                affect the output, but can affect the speed of the forward pass.
            max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
                (bin_size > 0); this is the maximum number of faces allowed within each
                bin. This should not affect the output values, but can affect
                the memory usage in the forward pass.
            perspective_correct: Bool, Whether to apply perspective correction when computing
                barycentric coordinates for pixels. This should be set to True if a perspective
                camera is used.
            clip_barycentric_coords: Whether, after any perspective correction is applied
                but before the depth is calculated (e.g. for z clipping),
                to "correct" a location outside the face (i.e. with a negative
                barycentric coordinate) to a position on the edge of the face.
            cull_backfaces: Bool, Whether to only rasterize mesh faces which are
                visible to the camera.  This assumes that vertices of
                front-facing triangles are ordered in an anti-clockwise
                fashion, and triangles that face away from the camera are
                in a clockwise order relative to the current view
                direction. NOTE: This will only work if the mesh faces are
                consistently defined with counter-clockwise ordering when
                viewed from the outside.
            z_clip_value: if not None, then triangles will be clipped (and possibly
                subdivided into smaller triangles) such that z >= z_clip_value.
                This avoids camera projections that go to infinity as z->0.
                Default is None as clipping affects rasterization speed and
                should only be turned on if explicitly needed.
                See clip.py for all the extra computation that is required.
            cull_to_frustum: if True, triangles outside the view frustum will be culled.
                Culling involves removing all faces which fall outside view frustum.
                Default is False so that it is turned on only when needed.

        Returns:
            4-element tuple containing

            - **pix_to_face**: LongTensor of shape
            (N, image_size, image_size, faces_per_pixel)
            giving the indices of the nearest faces at each pixel,
            sorted in ascending z-order.
            Concretely ``pix_to_face[n, y, x, k] = f`` means that
            ``faces_verts[f]`` is the kth closest face (in the z-direction)
            to pixel (y, x). Pixels that are hit by fewer than
            faces_per_pixel are padded with -1.
            - **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel)
            giving the NDC z-coordinates of the nearest faces at each pixel,
            sorted in ascending z-order.
            Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.
            - **barycentric**: FloatTensor of shape
            (N, image_size, image_size, faces_per_pixel, 3)
            giving the barycentric coordinates in NDC units of the
            nearest faces at each pixel, sorted in ascending z-order.
            Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives
            the barycentric coords for pixel (y, x) relative to the face
            defined by ``face_verts[f]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.
            - **pix_dists**: FloatTensor of shape
            (N, image_size, image_size, faces_per_pixel)
            giving the signed Euclidean distance (in NDC units) in the
            x/y plane of each point closest to the pixel. Concretely if
            ``pix_to_face[n, y, x, k] = f`` then ``pix_dists[n, y, x, k]`` is the
            squared distance between the pixel (y, x) and the face given
            by vertices ``face_verts[f]``. Pixels hit with fewer than
            ``faces_per_pixel`` are padded with -1.

            In the case that image_size is a tuple of (H, W) then the outputs
            will be of shape `(N, H, W, ...)`.
        """

        face_verts = meshes.face_verts
        mesh_to_face_first_idx = torch.tensor([0],device=face_verts.device, dtype=torch.int64)
        num_faces_per_mesh = torch.tensor([face_verts.size(0)],device=face_verts.device, dtype=torch.int64)


        if self.backend =='cuda':
            # In the case that H != W use the max image size to set the bin_size
            # to accommodate the num bins constraint in the coarse rasterizer.
            # If the ratio of H:W is large this might cause issues as the smaller
            # dimension will have fewer bins.
            # TODO: consider a better way of setting the bin size.
            im_size = parse_image_size(image_size)
            max_image_size = max(*im_size)

            clipped_faces_neighbor_idx = None


            if z_clip_value is not None or cull_to_frustum:
                # Cull faces outside the view frustum, and clip faces that are partially
                # behind the camera into the portion of the triangle in front of the
                # camera.  This may change the number of faces
                frustum = ClipFrustum(
                    left=-1,
                    right=1,
                    top=-1,
                    bottom=1,
                    perspective_correct=perspective_correct,
                    z_clip_value=z_clip_value,
                    cull=cull_to_frustum,
                )
                clipped_faces = clip_faces(
                    face_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum=frustum
                )
                face_verts = clipped_faces.face_verts
                mesh_to_face_first_idx = clipped_faces.mesh_to_face_first_idx
                num_faces_per_mesh = clipped_faces.num_faces_per_mesh

                # For case 4 clipped triangles (where a big triangle is split in two smaller triangles),
                # need the index of the neighboring clipped triangle as only one can be in
                # in the top K closest faces in the rasterization step.
                clipped_faces_neighbor_idx = clipped_faces.clipped_faces_neighbor_idx
            
            if clipped_faces_neighbor_idx is None:
                # Set to the default which is all -1s.
                clipped_faces_neighbor_idx = torch.full(
                    size=(face_verts.shape[0],),
                    fill_value=-1,
                    device=face_verts.device,
                    dtype=torch.int64,
                )
            

            # TODO: Choose naive vs coarse-to-fine based on mesh size and image size.
            if bin_size is None:
                # TODO better heuristics for bin size.
                if max_image_size <= 64:
                    bin_size = 8
                else:
                    # Heuristic based formula maps max_image_size -> bin_size as follows:
                    # max_image_size < 64 -> 8
                    # 16 < max_image_size < 256 -> 16
                    # 256 < max_image_size < 512 -> 32
                    # 512 < max_image_size < 1024 -> 64
                    # 1024 < max_image_size < 2048 -> 128
                    bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

            
            if bin_size != 0:
                # There is a limit on the number of faces per bin in the cuda kernel.
                faces_per_bin = 1 + (max_image_size - 1) // bin_size
                if faces_per_bin >= kMaxFacesPerBin:
                    raise ValueError(
                        "bin_size too small, number of faces per bin must be less than %d; got %d"
                        % (kMaxFacesPerBin, faces_per_bin)
                    )
            

            if max_faces_per_bin is None:
                max_faces_per_bin = int(max(10000, meshes.face_verts.size(0) / 5))
            

            face_bp_id = meshes.face_bp_idx


            bin_faces, elems_per_bin = RasterizeMeshesCoarseCuda(face_verts, mesh_to_face_first_idx, num_faces_per_mesh, im_size,blur_radius, bin_size,max_faces_per_bin)


            
            pix_to_face, zbuf, barycentric_coords, dists = RasterizeMeshesFineCuda(
                face_verts,
                bin_faces,
                elems_per_bin,
                clipped_faces_neighbor_idx,
                im_size,
                blur_radius,
                bin_size,
                faces_per_pixel,
                perspective_correct,
                clip_barycentric_coords,
                cull_backfaces
            )


            if z_clip_value is not None or cull_to_frustum:
                # If faces were clipped, map the rasterization result to be in terms of the
                # original unclipped faces.  This may involve converting barycentric
                # coordinates
                outputs = convert_clipped_rasterization_to_original_faces(
                    pix_to_face,
                    barycentric_coords,
                    # pyre-fixme[61]: `clipped_faces` may not be initialized here.
                    clipped_faces,
                )
                pix_to_face, barycentric_coords = outputs

        # Generate an image with the same size as pix_to_face
        bprimitive_image = torch.full_like(pix_to_face, -1, dtype=torch.int32)

        # Create a mask for valid face indices
        valid_mask = pix_to_face >= 0

        
        # Use advanced indexing to fill bprimitive_image with values from face_bp_id
        idx = pix_to_face[valid_mask]
        if idx.min()<0 or idx.max()>=face_bp_id.size(0):
            ipdb.set_trace()
        bprimitive_image[valid_mask] = face_bp_id[idx].int()
        bprimitive_image= bprimitive_image.squeeze()
        valid_mask = valid_mask.squeeze()


        outline_image, outline_coords = DetectOutlinePixelsCuda(bprimitive_image)


        return pix_to_face.squeeze(), zbuf, barycentric_coords, dists, {
            'outline_image':outline_image,
            'bprimitive_image': bprimitive_image,
            'valid_mask': valid_mask,
            'outline_coords': outline_coords
        }
