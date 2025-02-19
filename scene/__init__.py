import json
import os
import random
import torch
import numpy as np
from arguments import GroupParams
from model import BPrimitiveBase 
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud

from plyfile import PlyData
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return BasicPointCloud(points=positions, colors=None, normals=None)


class Scene:

    def __init__(self,
        args: GroupParams,
        bprimitive_object: BPrimitiveBase,
        load_iteration: int =  None,
        shuffle=True,
        resolution_scales=[1.0],
        gaussian_ori = None,
    ) -> None:
        """
        model_path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.bprimitive_object = bprimitive_object
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        colmap = False

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp, colmap_scale = args.colmap_scale)
            args.init_path = scene_info.ply_path
            colmap =  True
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)




        if not self.loaded_iter:
            if len(args.init_path)>2:
                if args.init_path.endswith(".obj"):
                    self.bprimitive_object.create_from_obj(args.init_path, self.cameras_extent)
                elif args.init_path.endswith(".ply"):
                    
                    point_cloud = fetchPly(args.init_path)

                    if colmap:
                        points = torch.tensor(point_cloud.points)*args.colmap_scale
                        print("using colmap point cloud!!, scale:", args.colmap_scale)
                    else:
                     
                        point_cloud.points[:, [1, 2]] = point_cloud.points[:, [2, 1]]
                        point_cloud.points[:, 1] *= -1
                        points = torch.tensor(point_cloud.points)

                    self.bprimitive_object.create_from_pc(points, args.init_triangle_size, self.cameras_extent, args.texture_resolution, args.texture_resolution_rest)
                else:
                    raise ValueError(f"got unsupport initialization: {args.init_path}")
            else:
                self.bprimitive_object.create_from_cube(self.cameras_extent)


        if gaussian_ori is not None:
            #pcd [N,3]
            if os.path.isfile(args.init_path):
                pcd = fetchPly(args.init_path)
                if not colmap:

                    pcd.points[:, [1, 2]] = pcd.points[:, [2, 1]]
                    pcd.points[:, 1] *= -1
                    points = pcd.points
                else:
                    points = pcd.points *args.colmap_scale
                
                
            else:
                points = np.random.uniform(low=-3, high=3, size=(100000, 3))

            gaussian_ori.create_from_pcd(points, self.cameras_extent)
            gaussian_ori.training_setup()

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]