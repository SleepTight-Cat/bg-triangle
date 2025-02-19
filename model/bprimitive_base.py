from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn
from utils.general_utils import get_expon_lr_func



class BPrimitiveBase(metaclass=ABCMeta):
    """
    Base class for the B-primitive.

    Args:
        order (int): Order of the B-primitive.
    """
    def __init__(self, order: int, sh_degree: int, optimizer_type: str = "default") -> None:
        super().__init__()

        self.num_feat_channel = 3

        self.num_primitives = 0
        self.order = order
        self.boundary_mode = -1

        self.texture_resolution = None

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.optimizer_type = optimizer_type
        self.control_point = torch.empty(0) # [s, num_control_points, 3]
        self.control_point_dc = torch.empty(0) # [s, num_control_points, 3]
        self.control_point_rest = torch.empty(0) # [s, num_control_points, 45]
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        self.single_features_rest = torch.empty(0)
        self.features_rotation = torch.empty(0)
        self.features_scaling = torch.empty(0)

        self.features_mlp = torch.empty(0)
        self.scaling = torch.empty(0)
        self.opacity = torch.empty(0)
        self.gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_activations()




    def capture(self):
        return (
            self.active_sh_degree,
            self.control_point,
            self.control_point_dc,
            self.control_point_rest,
            self.features_dc,
            self.features_rest,
            self.features_rotation,
            self.features_scaling, 
            self.scaling,
            self.single_features_rest,
            #self.opacity,
            #self.gradient_accum,
            #self.denom,
            #self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.boundary_mode
        )

    def restore(self, model_params: Any, training_args: Any) -> None:
        (
            self.active_sh_degree,
            self.control_point,
            self.control_point_dc,
            self.control_point_rest,
            self.features_dc,
            self.features_rest,
            self.features_rotation,
            self.features_scaling,
            self.scaling,
            self.single_features_rest,
            #self.opacity,
            #gradient_accum,
            #denom,
            #optimizer_state_dict,
            self.spatial_lr_scale,
            self.boundary_mode
        ) = model_params
        self.training_setup(training_args)
        self.num_primitives = self.control_point.shape[0]



    @classmethod
    def generate_ijk(cls, n: int) -> torch.Tensor:
        """i + j + k = n, 0 <= i, j, k <= n"""
        return torch.tensor([[i, j, n - i - j] for i in range(n, -1, -1) for j in range(n - i, -1, -1)])

    @abstractmethod
    def generate_regular_mesh(self, num_segments_per_edge: int) -> Any:
        raise NotImplementedError

    @property
    def num_control_points(self) -> int:
        return (self.order + 2) * (self.order + 1) // 2

    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    def setup_activations(self) -> None:
        """
        Activation functions for attributes.
        """
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.rotation_activation = torch.nn.functional.normalize

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

    def one_up_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity(), torch.ones_like(self.get_opacity())*0.6))
        self.opacity = opacities_new

    @abstractmethod
    def mode(self, id = 0):
        raise NotImplementedError


    def create_feature_texture(self, resolution =None, frest_resolution = 1):

        if self.texture_resolution is None and  resolution is not None:
            self.texture_resolution = resolution

        if resolution is None:
            resolution = self.texture_resolution

        self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, ((self.max_sh_degree + 1) ** 2 - 1) * 3, frest_resolution, frest_resolution, device="cuda"), requires_grad=True)
        self.features_scaling = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda")*-6, requires_grad=True)
        rots = torch.zeros(self.num_primitives, 4, resolution, resolution, device="cuda")
        rots[:,0] = 1
        self.features_rotation = nn.Parameter(rots, requires_grad=True)
        


    def training_setup(self, training_args: Any, lr_multiply: float = 1.0) -> None:
        self.percent_dense = training_args.percent_dense
        #------for split and prune------------------------------
        self.gradient_accum = torch.zeros((self.control_point.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.control_point.shape[0], 1), device="cuda")


        self.edge_accum = torch.zeros((self.control_point.shape[0], 1), device="cuda")
        self.denom_edge = torch.zeros((self.control_point.shape[0], 1), device="cuda")


        self.vis_accum = torch.zeros((self.control_point.shape[0], 1), device="cuda")
        self.denom_vis = torch.zeros((self.control_point.shape[0], 1), device="cuda")


        self.vis_map = torch.zeros((self.control_point.shape[0], 6,6), device="cuda")

        #-------------------------------------------------------

        # TODO: Rotation?
        # TODO: Optimizing scaling: No, but keep.
        if self.boundary_mode ==0:
            l = [
                {'params': [self.control_point], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_multiply, "name": "control_point"},
                {'params': [self.features_dc], 'lr': training_args.feature_lr_init * lr_multiply, "name": "f_dc"},
                {'params': [self.features_rest], 'lr': training_args.feature_lr_init / 20.0 * lr_multiply, "name": "f_rest"},
                #{'params': [self.scaling], 'lr': training_args.scaling_lr * lr_multiply, "name": "scaling"},
                {'params': [self.opacity], 'lr': training_args.opacity_lr * lr_multiply, "name": "opacity"},
            ]
        elif self.boundary_mode ==1:
            l = [
                {'params': [self.control_point], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_multiply, "name": "control_point"},

                {'params': [self.control_point_dc], 'lr':  training_args.feature_lr_init * lr_multiply, "name": "control_point_dc"},
                {'params': [self.control_point_rest], 'lr': training_args.feature_lr_init / 20.0  * lr_multiply, "name": "control_point_rest"},

                {'params': [self.features_dc], 'lr': training_args.feature_lr_init * lr_multiply, "name": "f_dc"},
                {'params': [self.features_rest], 'lr': training_args.feature_lr_init / 20.0 * lr_multiply, "name": "f_rest"},

                {'params': [self.single_features_rest], 'lr': training_args.feature_lr_init / 20.0 * lr_multiply, "name": "f_rest"},

                #{'params': [self.opacity], 'lr': training_args.opacity_lr * lr_multiply, "name": "opacity"},
            ]
            if training_args.newscaling:
                l.append({'params': [self.scaling], 'lr': training_args.scaling_lr * lr_multiply, "name": "scaling"})

        elif self.boundary_mode==2:
            l = [
                {'params': [self.features_dc], 'lr': training_args.feature_lr_init*2 * lr_multiply, "name": "f_dc"},
                {'params': [self.features_rest], 'lr': training_args.feature_lr_init *2/ 20.0 * lr_multiply, "name": "f_rest"},
                {'params': [self.features_scaling], 'lr': training_args.scaling_lr* lr_multiply, "name": "f_scale"},
                {'params': [self.features_rotation], 'lr': training_args.rotation_lr * lr_multiply, "name": "f_rotate"},
                #{'params': [self.single_features_rest_stage2], 'lr': training_args.feature_lr_init / 20.0 * lr_multiply, "name": "f_rest"},
                #{'params': [self.scaling], 'lr': training_args.scaling_lr * lr_multiply, "name": "scaling"},
                #{'params': [self.control_point], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_multiply, "name": "control_point"},
                #{'params': [self.control_point], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_multiply, "name": "control_point"},
                #{'params': [self.features_dc], 'lr': training_args.feature_lr * lr_multiply, "name": "f_dc"},
                #{'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0 * lr_multiply, "name": "f_rest"},
                #{'params': [self.scaling], 'lr': training_args.scaling_lr * lr_multiply, "name": "scaling"},
                #{'params': [self.opacity], 'lr': training_args.opacity_lr * lr_multiply, "name": "opacity"},
            ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.control_point_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * lr_multiply,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * lr_multiply,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        self.feature_scheduler_args = get_expon_lr_func(
            lr_init=training_args.feature_lr_init * lr_multiply,
            lr_final=training_args.feature_lr_final * self.spatial_lr_scale * lr_multiply,
            lr_delay_mult=training_args.feature_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration: int) -> None:
        """
        Learning rate scheduling per step.
        """
        lrs = {}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "control_point":
                lr = self.control_point_scheduler_args(iteration)
                param_group['lr'] = lr
                lrs[param_group["name"]] = param_group['lr']
            if param_group["name"] in ["f_dc", "control_point_dc", "f_scale", "f_rotate"]:
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
                lrs[param_group["name"]] = param_group['lr']

            if param_group["name"] in ["f_scale"]:
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr*3
                lrs[param_group["name"]] = param_group['lr']
            if param_group["name"] in ["f_rest", "control_point_rest"]:
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr / 20
                lrs[param_group["name"]] = param_group['lr']
        return lrs
