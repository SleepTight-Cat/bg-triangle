import math

import numpy as np

import torch
import torch.nn as nn

from .bprimitive_base import BPrimitiveBase
from utils.graphics_utils import BasicPointCloud
from utils.point_utils import generate_random_unit_triangles
import torch.nn.functional as F

from plyfile import PlyData
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return BasicPointCloud(points=positions, colors=None, normals=None)


class BPrimitiveBezier(BPrimitiveBase):
    """
    Triangle Bezier primitive.
    """
    def __init__(self, order: int, sh_degree: int, optimizer_type: str = "default") -> None:
        super().__init__(order, sh_degree, optimizer_type)

        self.ijk = self.generate_ijk(order) # [num_control_points, 3]
        self.coefficients = self.precompute_binomial_coefficients(order) # [n + 1, n + 1, n + 1]
        self.feature_ijk = self.generate_ijk(order) # [num_control_points, 3]
        self.feature_coefficients = self.precompute_binomial_coefficients(order) # [num_control_points, 3]


        # Regular mesh
        num_caches = 100
        self.face = self.generate_regular_face(num_caches) # [m * m, 3]
        self.bernstein_cache = {
            cache: self.compute_bernstein(self.coefficients, self.ijk, self.generate_uvw(cache)) for cache in range(1, num_caches + 1)
        } # [(m + 2) * (m + 1) // 2, num_control_points]

    def to(self, device: torch.device) -> "BPrimitiveBezier":
        self.ijk = self.ijk.to(device)
        self.coefficients = self.coefficients.to(device)
        self.feature_ijk = self.feature_ijk.to(device)
        self.feature_coefficients = self.feature_coefficients.to(device)

        self.face = self.face.to(device)
        self.bernstein_cache = {k: v.to(device) for k, v in self.bernstein_cache.items()}
        return self

    @classmethod
    def generate_uvw(cls, n: int) -> torch.Tensor:
        """
        Generate regular spaced UVW coordinates, where u + v + w = 1, 0 <= u, v, w <= 1.
        """
        return torch.tensor([[i / n, j / n, (n - i - j) / n] for i in range(n, -1, -1) for j in range(n - i, -1, -1)])

    def precompute_binomial_coefficients(self, n: int) -> torch.Tensor:
        """
        n! / (i! * j! * k!)
        """
        coefficients = torch.zeros(n + 1, n + 1, n + 1)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                coefficients[i, j, k] = math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
        return coefficients

    def compute_bernstein(self, coefficients: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * u^{i} * v^{j} * w^{k}
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [N, 1]
        return coefficients[i, j, k] * (u ** i) * (v ** j) * (w ** k) # [N, num_control_points]

    def compute_d_bernstein_d_u(self, coefficients: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * (u^{i-1} * v^{j} * w^{k} - u^{i} * v^{j} * w^{k})
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [N, 1]
        return coefficients[i, j, k] * (
            torch.where(i > 0, i * u ** (i - 1), torch.zeros_like(u)) * (v ** j) * (w ** k) - (u ** i) * (v ** j) * torch.where(k > 0, k * w ** (k - 1), torch.zeros_like(w))
        )

    def compute_d_bernstein_d_v(self, coefficients: torch.Tensor, ijk: torch.LongTensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        n! / (i! * j! * k!) * (u^{i} * v^{j-1} * w^{k} - u^{i} * v^{j} * w^{k})
        """
        i, j, k = ijk.unsqueeze(-3).unbind(-1) # [1, num_control_points]
        u, v, w = uvw.unsqueeze(-2).unbind(-1) # [N, 1]
        return coefficients[i, j, k] * (
            (u ** i) * torch.where(j > 0, j * v ** (j - 1), torch.zeros_like(v)) * (w ** k) - (u ** i) * (v ** j) * torch.where(k > 0, k * w ** (k - 1), torch.zeros_like(w))
        )

    def generate_regular_mesh(self, num_segments_per_edge: int, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate regular spaced triangle mesh.

        Args:
            num_segments_per_edge (`int`): Number of segments per bprimitive edge.
                Them each bprimitive generates `(m + 1) * (m + 2) // 2` vertices and `m * m` faces.
            mask (`torch.Tensor`): Mask for the control points, shape [s].
        """
        if mask is None:
            vertices = torch.einsum("scd, mc -> smd", self.control_point, self.bernstein_cache[num_segments_per_edge]) # [s, (m + 2) * (m + 1) // 2, 3]
            num_primitives = self.num_primitives
        else:
            vertices = torch.einsum("scd, mc -> smd", self.control_point[mask], self.bernstein_cache[num_segments_per_edge]) # [s, (m + 2) * (m + 1) // 2, 3]
            num_primitives = vertices.size(0)
        faces = self.face[:num_segments_per_edge * num_segments_per_edge].unsqueeze(0).expand(num_primitives, -1, -1) \
              + torch.arange(num_primitives, device=self.face.device).reshape(num_primitives, 1, 1) * ((num_segments_per_edge + 1) * (num_segments_per_edge + 2) // 2) # [s, m * m, 3]
        return vertices, faces

    def generate_regular_face(self, m: int) -> torch.Tensor:
        """
        Generate all faces indices for various `m`.
        """
        face = []
        for i in range(m):
            l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
            for j in range(l, r - 1):
                face.append([j, j + i + 1, j + i + 2])
                face.append([j, j + i + 2, j + 1])
            face.append([r - 1, r + i, r + i + 1])
        return torch.tensor(face) # [m * m, 3]

    def generate_from_triangles(self, trangles: torch.Tensor) -> torch.Tensor:
        """
        Generate control points from trangles.

        Args:
            trangles (torch.Tensor): Trangles vertices, which can be regarded as order 1 B-primitive, shape [N, 3, 3].

        Returns:
            control_points (torch.Tensor): Shape [N, num_control_points, 3].
        """
        # Here, ijk = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # [3, 3]
        uvw = self.generate_uvw(self.order).to(trangles.device) # [num_control_points, 3]
        control_points = torch.einsum("ncd, mc -> nmd", trangles, uvw) # [N, num_control_points, 3]
        return control_points

    def evaluate(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        bernstein = self.compute_bernstein(self.coefficients, self.ijk, uvw) # [N, num_control_points]
        vertices = torch.einsum("ncd, nc -> nd", self.control_point[bprimitive_id], bernstein) # [N, 3]
        return vertices

    def evaluate_dc(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        bernstein = self.compute_bernstein(self.feature_coefficients, self.feature_ijk, uvw) # [N, num_control_points]
        vertices = torch.einsum("ncd, nc -> nd", self.control_point_dc[bprimitive_id], bernstein) # [N, 3]
        return vertices

    def evaluate_rest(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        bernstein = self.compute_bernstein(self.feature_coefficients, self.feature_ijk, uvw) # [N, num_control_points]
        vertices = torch.einsum("ncd, nc -> nd", self.control_point_rest[bprimitive_id], bernstein) # [N, 45]
        return vertices

    def evaluate_u_derivative(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            d_vertices_d_u (torch.Tensor): Derivative of the vertices w.r.t. U [N, 3].
        """
        d_bernstein_d_u = self.compute_d_bernstein_d_u(self.coefficients, self.ijk, uvw)
        d_vertices_d_u = torch.einsum("ncd, nc -> nd", self.control_point[bprimitive_id], d_bernstein_d_u)
        return d_vertices_d_u

    def evaluate_v_derivative(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            d_vertices_d_v (torch.Tensor): Derivative of the vertices w.r.t. V [N, 3].
        """
        d_bernstein_d_v = self.compute_d_bernstein_d_v(self.coefficients, self.ijk, uvw)
        d_vertices_d_v = torch.einsum("ncd, nc -> nd", self.control_point[bprimitive_id], d_bernstein_d_v)
        return d_vertices_d_v

    def evaluate_normal(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            normals (torch.Tensor): Normals [N, 3].
        """
        d_vertices_d_u = self.evaluate_u_derivative(bprimitive_id, uvw)
        d_vertices_d_v = self.evaluate_v_derivative(bprimitive_id, uvw)
        normals = torch.cross(d_vertices_d_u, d_vertices_d_v, dim=-1)
        return normals

    def clone(self) -> "BPrimitiveBezier":
        bprimitive = BPrimitiveBezier(self.order, self.max_sh_degree)

        bprimitive.control_points.data = self.control_point.data.clone()
        bprimitive.features_dc.data = self.features_dc.data.clone()
        bprimitive.features_rest.data = self.features_rest.data.clone()
        bprimitive.features_mlp.data = self.features_mlp.data.clone()
        bprimitive.scaling.data = self.scaling.data.clone()
        bprimitive.opacity.data = self.opacity.data.clone()
        bprimitive.gradient_accum.data = self.gradient_accum.data.clone()
        bprimitive.denom.data = self.denom.data.clone()
        return bprimitive

    def mode(self, id = 0):
        '''
        mode:
        0 -> non_boundary_mode
        1 -> boundary_mode
        '''

        if  self.boundary_mode==id:
            return False
        self.boundary_mode =  id


        return True

    def create_from_cube(self, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        from model import experimental_cube_get_bprimitive_gaussian
        control_points = experimental_cube_get_bprimitive_gaussian(
            num_samples_per_cube_edge=2,
            order=self.order
        ).cuda()
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        resolution = 2

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        print("Shape of control points points at CUBE initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]
        # self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 1, 3, device="cuda"), requires_grad=True)
        # self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution)
        #self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 1, 3, device="cuda"), requires_grad=True)
        self.single_features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)
        #self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda"), requires_grad=True)
        #self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, ((self.max_sh_degree + 1) ** 2 - 1) * 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")


    def create_from_pc(self, points, init_triangle_size: float, spatial_lr_scale: float, resolution: int, resolution_res: int = 1) -> None:
        self.spatial_lr_scale = spatial_lr_scale

    
        points = points.unsqueeze(1).repeat(1,3,1)
        idx = torch.randperm(points.shape[0])[:1000] # Assume initial_pc larger than 1000
        points = points[idx]

        triangles  = generate_random_unit_triangles(points.size(0))*init_triangle_size

        triangles = triangles + points

        control_points = torch.from_numpy(triangles.detach().cpu().numpy()).float().cuda()
        control_points = self.generate_from_triangles(control_points)
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution, resolution_res)

        self.single_features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)
       
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")





    def create_from_obj(self, init_path: str, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        # Load obj file
        import trimesh
        mesh = trimesh.load_mesh(init_path, force='mesh')

        vertices = mesh.vertices
        faces = mesh.faces
        # 对于blender 导出的模型需要使用以下变换！！！
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices[:, 1] *= -1

        resolution = 4

        control_points = torch.from_numpy(vertices[faces]).float().cuda()
        control_points = self.generate_from_triangles(control_points)
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution)
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")

    def create_from_random(self, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        from model import experimental_triangle_get_bprimitive_random

        bprimitive_object = experimental_triangle_get_bprimitive_random(50,self.order)

        resolution = 3


        self.control_point = nn.Parameter(bprimitive_object.control_points.clone().cuda(), requires_grad=True)
        print("Shape of control points points at initialization: ", self.control_point.shape)
        self.num_primitives = self.control_point.shape[0]
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, ((self.max_sh_degree + 1) ** 2 - 1) * 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), 0.004, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")


    #----------------- Version 1 ---------------------------

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, bprimitive_ids):
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)

        src = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        src[src>1e-6] = 1e-9

        tmp = torch.zeros_like(self.gradient_accum).scatter_add(
            dim=0,
            index=bprimitive_ids.unsqueeze(-1), 
            src=src
        )
        tmp[unique_bprimitive_ids, 0] /= counts
        self.gradient_accum += tmp
        self.denom[unique_bprimitive_ids] += 1

    def split_bprimitive(self, edge_threshold = 5, grad_threshold = 5e-4, area_threshold = 0.00003):
        metrics = self.gradient_accum / (self.denom + 1)
        mask = metrics >= grad_threshold


        metrics = self.edge_accum / (self.denom_edge + 1)
        mask = mask | (metrics >= edge_threshold)


        mask = mask.squeeze(1)

        vertices, faces = self.generate_regular_mesh(2)
        vertices= vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        v0 = vertices[faces[:, 0]]  # A
        v1 = vertices[faces[:, 1]]  # B
        v2 = vertices[faces[:, 2]]  # C

        AB = v1 - v0
        AC = v2 - v0
        cross_product = torch.cross(AB, AC, dim=-1)
        areas = 0.5 * torch.norm(cross_product, dim=-1)
        areas = areas.reshape(-1,4).sum(dim=1)


        mask = mask & (areas > area_threshold*2)


        num_primitive_to_split = mask.sum()

        if num_primitive_to_split != 0:
            split_points = self.generate_uvw(self.order).repeat(num_primitive_to_split, 1).cuda()
            bids = torch.arange(0, mask.size(0), device="cuda")
            bids = torch.repeat_interleave(bids[mask], self.num_control_points)

            new_trangle_points = self.evaluate(bids, split_points)
            new_trangle_points = new_trangle_points.reshape(num_primitive_to_split,-1,3)


            vertices2, faces2 = self.generate_regular_mesh(2, mask)
            vertices2 = vertices2.reshape(-1, 3)
            faces2 = faces2.reshape(-1, 3)
            new_trangles = vertices2[faces2]





            new_control_points = self.generate_from_triangles(new_trangles)

            if new_control_points.size(0)+(~mask).sum() < 202000:  # maximum number of primitives
                center = new_control_points.mean(dim=1).unsqueeze(1)
                new_control_points = (new_control_points-center)*1.1+center
                self.update_primitives_add(new_control_points, mask)
                print(f'New primitives: {new_control_points.size(0)}')

    def update_primitives_add(self, new_control_points, primitives_to_remove):
        self.control_point = nn.Parameter(torch.cat([
            self.control_point.data[~primitives_to_remove], new_control_points
        ], dim=0))

        self.control_point_dc = nn.Parameter(torch.cat([
            self.control_point_dc.data[~primitives_to_remove], self.control_point_dc.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0))


        self.control_point_rest = nn.Parameter(torch.cat([
            self.control_point_rest.data[~primitives_to_remove], self.control_point_rest.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0))

        
        self.num_primitives = self.control_point.size(0)

        self.gradient_accum = torch.zeros((self.control_point.size(0), 1), device="cuda")
        self.denom = torch.zeros((self.control_point.size(0), 1), device="cuda")


        def resize_split(X):
            
            to_process = X  
            h = X.size(-1)

            processed = None

            
            if to_process.shape[0] == 0:
                pass
            else:
               
                h_half = h // 2
                top_left = to_process[:, :, :h_half, :h_half]  # (M, 3, h/2, h/2)
                top_right = to_process[:, :, :h_half, h_half:]  # (M, 3, h/2, h/2)
                bottom_left = to_process[:, :, h_half:, :h_half]  # (M, 3, h/2, h/2)
                bottom_right = to_process[:, :, h_half:, h_half:]  # (M, 3, h/2, h/2)

                
                top_left_upscaled = F.interpolate(top_left, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                top_right_upscaled = F.interpolate(top_right, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                bottom_left_upscaled = F.interpolate(bottom_left, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                bottom_right_upscaled = F.interpolate(bottom_right, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)

                processed = torch.stack((top_left_upscaled, bottom_left_upscaled, top_left_upscaled,top_left_upscaled ), dim=1)  # (M, 4, 3, h, h)

                processed = processed.view(-1, X.size(1), h, h)  # (4 * M, 3, h, h)
            return processed




        self.single_features_rest =  nn.Parameter(torch.cat([
            self.single_features_rest.data[~primitives_to_remove], self.single_features_rest.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())



        self.scaling = nn.Parameter(torch.cat([
            self.scaling.data[~primitives_to_remove], self.scaling.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())
        self.opacity = nn.Parameter(torch.cat([
            self.opacity.data[~primitives_to_remove], self.opacity.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())

        self.vis_accum = torch.cat([
                        self.vis_accum[~primitives_to_remove], self.vis_accum[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)
        self.denom_vis = torch.cat([
                        self.denom_vis[~primitives_to_remove], self.denom_vis[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)

        self.vis_map = torch.cat([
                        self.vis_map[~primitives_to_remove], self.vis_map[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)



        


    def update_primitives_del(self,  primitives_to_remove):
        self.control_point = nn.Parameter(self.control_point.data[~primitives_to_remove])

        self.control_point_dc = nn.Parameter(self.control_point_dc.data[~primitives_to_remove])
        self.control_point_rest = nn.Parameter(self.control_point_rest.data[~primitives_to_remove])
        
        self.num_primitives = self.control_point.size(0)

        self.gradient_accum = self.gradient_accum[~primitives_to_remove]
        self.denom = self.denom[~primitives_to_remove]

        self.single_features_rest = nn.Parameter(self.single_features_rest.data[~primitives_to_remove])


        self.scaling = nn.Parameter(self.scaling.data[~primitives_to_remove].contiguous())
        self.opacity = nn.Parameter(self.opacity.data[~primitives_to_remove].contiguous())

    def prune(self, min_opacity, vis_threshold = 0.02, area_threshold = 0.00003):
        # Prune primitives with small opacity
        mask = self.opacity.data < min_opacity

        metrics = self.vis_accum / (self.denom_vis + 1)
        mask = mask | ( metrics < vis_threshold)

        mask = mask.squeeze(1)

        # Prune primitives with small areas
        vertices, faces = self.generate_regular_mesh(2)
        vertices= vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        v0 = vertices[faces[:, 0]]  # A
        v1 = vertices[faces[:, 1]]  # B
        v2 = vertices[faces[:, 2]]  # C

        AB = v1 - v0
        AC = v2 - v0
        cross_product = torch.cross(AB, AC, dim=-1)
        areas = 0.5 * torch.norm(cross_product, dim=-1)
        areas = areas.reshape(-1,4).sum(dim=1)

        # Prune primitives with large aspect ratios
        vertices, faces = self.generate_regular_mesh(2)
        bbox_max = vertices.max(dim=1)[0]
        bbox_min = vertices.min(dim=1)[0]

        lengths = bbox_max - bbox_min
        lengths = lengths.sort(dim=1)[0]

        ratio = lengths[:, 2] / lengths[:, 1]

        # Update mask
        area_mask = areas < area_threshold
        ratio_mask = (ratio > 20)
        seen_mask = (self.vis_map.view(-1,36)>0).sum(dim=1) <8

        masks = mask | area_mask | ratio_mask | seen_mask
        print(f'Prune {masks.sum()} primitives, {mask.sum().item()} visibility, {area_mask.sum().item()} areas, {ratio_mask.sum()} ratio, {seen_mask.sum()} unseen')

        self.update_primitives_del(masks)

    @torch.no_grad()
    def densify_and_prune(self, edge_threshold = 5, grad_threshold = 5e-4, vis_threshold = 0.02, area_threshold = 0.00003):

        self.prune(0.3, vis_threshold =vis_threshold, area_threshold = area_threshold)
        self.num_primitives = self.control_point.shape[0]
        print(f'Total #Primitives: {self.num_primitives}')
        if(self.num_primitives > 2000): ###
            return
        self.split_bprimitive(edge_threshold = edge_threshold,  grad_threshold = grad_threshold, area_threshold = area_threshold)


    #----------------- Version 2 ---------------------------

    @torch.no_grad()
    def add_densification_stats_v2(self, edge_map, bprimitive_image):

        mask = bprimitive_image != -1
        bprimitive_ids = bprimitive_image[mask]
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)


        src = edge_map.mean(dim=0)
        src = src[mask]





        tmp = torch.zeros_like(self.gradient_accum).scatter_add(
            dim=0,
            index=bprimitive_ids.unsqueeze(-1).long(), 
            src=src.unsqueeze(-1)
        )

        #mp[unique_bprimitive_ids, 0] /= counts
        self.edge_accum += tmp
        self.denom_edge[unique_bprimitive_ids] += 1


    @torch.no_grad()
    def add_densification_stats_v3(self):
        grads = self.control_point.grad

        grads = grads.norm(dim=-1).sum(dim=-1).unsqueeze(-1)
        self.gradient_accum += grads
        self.denom =  self.denom+ 1


    @torch.no_grad()
    def add_densification_stats_v4(self, bprimitive_image):

        mask = bprimitive_image != -1
        bprimitive_ids = bprimitive_image[mask]
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)

        
        self.vis_accum[unique_bprimitive_ids] += 1
        self.denom_vis +=  1