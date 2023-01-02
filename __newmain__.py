#
from __future__ import absolute_import, division, print_function
import glob
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch

from torchvision import transforms

from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder
#
import argparse
from pathlib import Path
import pickle
import time
from typing import Dict, List
import os
import sys
#
import matplotlib.pyplot as plt
import pdb
#
from PIL import Image
import cv2 as cv
import kornia
from logzero import logger
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from edam.dataset import hamlyn
from edam.optimization.frame import create_frame
from edam.optimization.pose_estimation import PoseEstimation
from edam.optimization.utils_frame import synthetise_image_and_error
from edam.utils.file import list_files
from edam.utils.image.convertions import (
    numpy_array_to_pilimage,
    pilimage_to_numpy_array,
)
from edam.utils.image.pilimage import (
    pilimage_h_concat,
    pilimage_rgb_to_bgr,
    pilimage_v_concat,
)
from edam.utils.depth import depth_to_color
from edam.utils.LineMesh import LineMesh
from edam.utils.parser import txt_to_nparray


def parse_args() -> argparse.Namespace:
    """Returns the ArgumentParser of this app.

    Returns:
        argparse.Namespace -- Arguments
    """
    parser = argparse.ArgumentParser(
        description="Shows the images from Hamlyn Dataset"
    )
    parser.add_argument(
        "-i",
        "--input_root_directory",
        type=str,
        required=True,
        help="Root directory where scans are find. E.G. path/to/hamlyn_tracking_test_data",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device in where to run the optimization E.G. cpu, cuda:0, cuda:1",
    )
    parser.add_argument(
        "-fr",
        "--ratio_frame_keyframe",
        type=int,
        default=2,
        help="Number of frames until a new keyframes.",
    )
    parser.add_argument(
        "-s",
        "--start_scene",
        type=int,
        default=0,
        help="Select the start scene",
    )
    parser.add_argument(
        "-o",
        "--folder_output",
        type=str,
        default="results",
        help="Folder where odometries are saved.",
    )
    parser.add_argument(
        "-st",
        "--scales_tracking",
        type=int,
        default=2,
        help="Number of floors of the pyramid.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = args.folder_output
    device_name = args.device
    list_scenes = list_files(args.input_root_directory)

    done = False
    scene_number = args.start_scene
    frame_number = 0
    scene_info = None

    estimated_pose = np.eye(4)
    automode = False

    pe = PoseEstimation()
    keyframetime=0
    nkeyframetime=0
    createmeshtime=0
    keyframe_image = []

    vis_3d = o3d.visualization.Visualizer()
    vis_3d.create_window("HAMLYN3D", 1920 // 2, 1080 // 2)
#
    folder_to_save_results = Path(args.folder_output)
    if not (folder_to_save_results.exists()):
        folder_to_save_results.mkdir(parents=True, exist_ok=True)
    #
    #output_path = folder_to_save_results / (time.strftime("%Y%m%d-%H%M%S") + ".pkl")
    
    #
    scales = args.scales_tracking

    while not done:
        # -- Load info for the scene if it has not been loaded yet.
        begin = time.time()
        if scene_info is None:
            logger.info("Reseting scene.")
            scene = list_scenes[scene_number]
            args.image_path=os.path.join(scene, 'color')
            args.output_path=os.path.join(folder_to_save_results, 'depth')
            scene_info = hamlyn.load_scene_files(scene)
            frame_number = 0
            points = [np.array([0, 0, 0])]
            

        # Get current camera
        ctr = vis_3d.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        # -- Open data
        (
            rgb_image_registered,
            depth_np,
            k_depth,
            h_d,
            w_d,
        ) = load_hamlyn_frame(scene_info, frame_number)

        depth_image = numpy_array_to_pilimage(
            (
                depth_to_color(
                    depth_np,
                    cmap="jet",
                    max_depth=np.max(depth_np),
                    min_depth=np.min(depth_np),
                )
            ).astype(np.uint8)
        )
	
        gray = cv.cvtColor(
            pilimage_to_numpy_array(rgb_image_registered), cv.COLOR_BGR2GRAY
        )

        new_frame = create_frame(
            c_pose_w=np.linalg.inv(estimated_pose),
            c_pose_w_gt=None,
            gray_image=gray,
            rgbimage=None,
            depth=depth_np,
            k=k_depth.numpy().reshape(3, 3),
            idx=frame_number,
            ref_camera=(frame_number == 0),
            scales=scales,
            code_size=128,
            device_name=device_name,
            uncertainty=None,
        )

        # Keyframe updating
        if frame_number == 0:
            new_frame.modify_pose(c_pose_w=np.linalg.inv(np.eye(4)))

        if frame_number % args.ratio_frame_keyframe == 0:
            logger.info("KEYFRAME INSERTED")
            new_frame_ = create_frame(
                c_pose_w=new_frame.c_pose_w,
                c_pose_w_gt=None,
                gray_image=gray,
                rgbimage=None,
                depth=depth_np,
                k=k_depth.numpy().reshape(3, 3),
                idx=frame_number,
                ref_camera=True,
                scales=1,
                code_size=128,
                device_name=device_name,
                uncertainty=None,
            )
            if not (frame_number == 0):
                pe.run(new_frame_, True)

            pe.set_ref_keyframe(new_frame_)
            ref_kf = pe.reference_keyframe
            keyframe_image = rgb_image_registered.copy()
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    cv.cvtColor(
                        pilimage_to_numpy_array(keyframe_image), cv.COLOR_BGR2RGB
                    )
                ),
                o3d.geometry.Image(depth_np * 1000),
                convert_rgb_to_intensity=False,
            )
#
            #pdb.set_trace()
#
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                w_d,
                h_d,
                k_depth[0, 0, 0],  # fx
                k_depth[0, 1, 1],  # fy
                k_depth[0, 0, 2],  # cx
                k_depth[0, 1, 2],  # cy
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic, new_frame.c_pose_w,
            )
            #end = time.time()
            #time_keyframetime_image = round((end - begin) * 1000)
            #keyframetime+=time_keyframetime_image
            #print("   Computing keyframe time {}ms".format(time_keyframetime_image))

        else:
            #end = time.time()
            #time_nkeyframetime_image = round((end - begin) * 1000)
            #nkeyframetime+=time_nkeyframetime_image
            #print("   Computing not keyframe time {}ms".format(time_nkeyframetime_image))
            pe.run(new_frame, True)
            ref_kf = pe.reference_keyframe



        frame = pilimage_v_concat([pilimage_h_concat([rgb_image_registered.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR)]),])

        estimated_pose = np.linalg.inv(new_frame.c_pose_w)


        logger.info(
            f"frame {frame_number:d}"
        )
        logger.info(f"")
#
        poses = new_frame.c_pose_w.copy()
        frame_numbers = frame_number
        #begin=time.time()
        if frame_number == 0:
            #create_frames=[1,2,3,4,5,10,20,30,40,50,1057]
            cam.extrinsic = np.eye(4)
            rgbd_images = []
            #points = []
            #linemesh=[]
            tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.001,
                sdf_trunc=0.005,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
            with open(os.path.join(folder_to_save_results, 'poses.log'), 'w') as traj:
                traj.write(f"0 0 {frame_number}\n"
                           f"{poses[0, 0]} {poses[0, 1]} {poses[0, 2]} {poses[0, 3]}\n"
                           f"{poses[1, 0]} {poses[1, 1]} {poses[1, 2]} {poses[1, 3]}\n"
                           f"{poses[2, 0]} {poses[2, 1]} {poses[2, 2]} {poses[2, 3]}\n"
                           f"{poses[3, 0]} {poses[3, 1]} {poses[3, 2]} {poses[3, 3]}\n"
                           )

        # Create poses.log
        else:
            with open(os.path.join(folder_to_save_results, 'poses.log'), 'a') as traj:
                traj.write(f"0 0 {frame_number}\n"
                           f"{poses[0, 0]} {poses[0, 1]} {poses[0, 2]} {poses[0, 3]}\n"
                           f"{poses[1, 0]} {poses[1, 1]} {poses[1, 2]} {poses[1, 3]}\n"
                           f"{poses[2, 0]} {poses[2, 1]} {poses[2, 2]} {poses[2, 3]}\n"
                           f"{poses[3, 0]} {poses[3, 1]} {poses[3, 2]} {poses[3, 3]}\n"
                           )

        trajectory = o3d.io.read_pinhole_camera_trajectory(os.path.join(root, 'poses.log'))
        _, _, intrinsic = load_frame(root, frame_number)
        print("Changing intrinsics of the {:d}-th image.".format(frame_number))
        trajectory.parameters[frame_number].intrinsic = intrinsic

        rgbd_images.append(compute_rgbd_images(root, frame_number))
        print("Integrate {:d}-th image into the volume.".format(frame_number))
        tsdf.integrate(rgbd_images[frame_number], intrinsic, np.linalg.inv(trajectory.parameters[frame_number].extrinsic))
        mesh = tsdf.extract_triangle_mesh()
        #end = time.time()
        #time_createmeshtime_image = round((end - begin) * 1000)
        #createmeshtime+=time_createmeshtime_image
        vis_3d.clear_geometries()
        vis_3d.add_geometry(mesh)

        '''
        if frame_number > 0:
            pose = np.linalg.inv(poses)
            position = pose[:, 3]
            points.append(position[:3])
            line_mesh = LineMesh(points, radius=0.0005)
            line_mesh_geoms = line_mesh.cylinder_segments
            vis_3d.add_geometry(*line_mesh_geoms)
            print(line_mesh_geoms)
            del points[0]
        '''
        
        #print("   Computing mesh time {}ms".format(time_createmeshtime_image))
        # Draw trajectory
        #if frame_number in create_frames:
            #o3d.io.write_triangle_mesh(os.path.join(folder_to_save_results, 'mesh{}'.format(frame_number)+'.ply'),
                                       #mesh,
                                       #write_ascii=True)
        """
        if frame_number % args.ratio_frame_keyframe == 0:
            f=open(os.path.join(folder_to_save_results, 'time.txt'),'a')
            f.write(f"one keyframe time = {time_keyframetime_image}\n"
                    f"create one mesh time = {time_createmeshtime_image}\n")
            f.close()

        else:
            f=open(os.path.join(folder_to_save_results, 'time.txt'),'a')
            f.write(f"one not keyframe time = {time_nkeyframetime_image}\n"
                    f"create one mesh time = {time_createmeshtime_image}\n")
            f.close()

        if frame_number >= len(scene_info["list_color_images"])-1:
            f=open(os.path.join(folder_to_save_results, 'time.txt'),'a')
            f.write(f"keyframe total time = {keyframetime}\n"
                    f"not keyframe total time = {nkeyframetime}\n"
                    f"create mesh total time = {createmeshtime}\n")
            f.close()
        """
        if frame_number >= len(scene_info["list_color_images"])-1:
            o3d.io.write_pinhole_camera_trajectory(os.path.join(folder_to_save_results, 'trajectory.log'), trajectory)
            o3d.io.write_triangle_mesh(os.path.join(folder_to_save_results, 'mesh.ply'),
                                       mesh,
                                       write_ascii=True)
            os.remove(os.path.join(root, 'poses.log'))

#
        #pdb.set_trace()

        camera_ = o3d.camera.PinholeCameraParameters()
        camera_.intrinsic = cam.intrinsic
        camera_.extrinsic = cam.extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_)

        cv.imshow("HAMLYN", pilimage_to_numpy_array(frame))
        end = time.time()
        time_keyframetime_image = round((end - begin) * 1000)
        print("   Computing total time {}ms".format(time_keyframetime_image))

        while cv.getWindowProperty("HAMLYN", 0) >= 0:
            if frame_number >= len(scene_info["list_color_images"])-1:
                automode = False
            vis_3d.poll_events()
            vis_3d.update_renderer()
            key = cv.waitKey(1) & 0xFF
            
            
            if automode:
                if key == ord("s"):
                   automode = False
                   #print("*******"+str(len(scene_info["list_color_images"])))
                frame_number = (frame_number + 1) #% len(scene_info["list_color_images"])
                break
            if key == ord("a"):
                automode = True
                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 1) #% len(scene_info["list_color_images"])
                break
            if key == ord("n"):

                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 1) #% len(scene_info["list_color_images"])
                break
            if key == ord("j"):
                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 10) #% len(scene_info["list_color_images"])
                if frame_number >= len(scene_info["list_color_images"]):
                    frame_number = (frame_number - 10)
                    #print("**********Out of Range!!!!!")
                    break
                break
            elif key == ord("p"):
                logger.info(help_functions()[chr(key)])
                frame_number = frame_number - 1
                if frame_number < 0:
                    frame_number = 0
                break
            elif key == ord(" "):
                logger.info(help_functions()[chr(key)])
                scene_number = (scene_number + 1) % len(list_scenes)
                pe = PoseEstimation()
                frame_number = 0
                scene_info = None
                break
            elif key == ord("\x08"):
                logger.info(help_functions()[chr(key)])
                scene_number = scene_number - 1
                pe = PoseEstimation()
                if scene_number < 0:
                    scene_number = len(list_scenes) - 1
                frame_number = 0
                scene_info = None
                break
            elif key == ord("h"):
                logger.info(help_functions()[chr(key)])
                print_help()
            elif key == ord("q"):
                logger.info(help_functions()[chr(key)])
                posesfile=Path(os.path.join(root, 'poses.log'))
                if (posesfile.exists()):
                    os.remove(posesfile)
                done = True
                break
            elif key != 255:
                logger.info(f"Unkown command, {chr(key)}")
                print("Use 'h' to print help and 'q' to quit.")


def load_hamlyn_frame(scene_info: Dict[str, List[str]], frame_number: int):
    """Function to load the scene and frame from the Hamlyn dataset,

    Args:
        scene_info ([type]): [description]
        frame_number (int): [description]

    Returns:
        [type]: [description]
    """
    # -- Open the rgb image.
    rgb_path = scene_info["list_color_images"][frame_number]
    rgb_image = pilimage_rgb_to_bgr(Image.open(rgb_path))

    # -- Open the depth image.
    depth_path = scene_info["list_depth_images"][frame_number]
    depth_np = cv.imread(depth_path, cv.IMREAD_ANYDEPTH).astype(np.float32) / 1000

    # -- Get the camera matrices.
    k_depth: np.ndarray = scene_info["intrinsics"][:3, :3]  # type: ignore

    # -- Transform into tensors.
    k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3

    h_d, w_d = depth_np.shape

    return (
        rgb_image,
        depth_np,
        k_depth,
        h_d,
        w_d,
    )


def help_functions():
    helper = {}
    helper["a"] = "Autoplay"
    helper["s"] = "Stop Autoplay"
    helper["n"] = "Next Image"
    helper["j"] = "Next Image x10"
    helper["p"] = "Previous Image"
    helper["\x08"] = "(Backslash) Previous Scene"
    helper[" "] = "(Space bar) Next Scene"
    helper["h"] = "Print help"
    helper["q"] = "Quit program"
    return helper


def print_help():
    logger.info("Printing help:")
    for k, v in help_functions().items():
        print(k, ":\t", v)


#from volumetric fusion
def compute_rgbd_images(root, frame_numbers):
    print("Store {:d}-th image into rgbd_images.".format(frame_numbers))
    color, depth, _ = load_frame(root, frame_numbers)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        o3d.geometry.Image(depth),
        depth_trunc=30000.0,
        depth_scale=1.0,
        convert_rgb_to_intensity=False
    )

    return rgbd


def load_frame(root, frame_number: int):
    color = o3d.io.read_image(os.path.join(root, "color", "{:010d}.jpg".format(frame_number)))
    depth = cv.imread(os.path.join(root, "depth", "{:010d}.png".format(frame_number)), cv.IMREAD_ANYDEPTH)\
                .astype(np.float32) / 1000  # meters
    intrinsics = txt_to_nparray(os.path.join(root, "intrinsics.txt"))
    k: np.ndarray = intrinsics[:3, :3]  # type: ignore

    # -- Transform into tensors.
    k = kornia.utils.image_to_tensor(k, keepdim=False).squeeze(1)  # Bx3x3

    h, w = depth.shape

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w,
        h,
        k[0, 0, 0],  # fx
        k[0, 1, 1],  # fy
        k[0, 0, 2],  # cx
        k[0, 1, 2],  # cy
    )

    return color, depth, intrinsic
#
if __name__ == "__main__":
    main()
