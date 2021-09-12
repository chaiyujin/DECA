# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
from collections import OrderedDict
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.io import imread
from tqdm import tqdm

from decalib.models.FLAME import FLAME, FLAMETex
from decalib.utils import lossfunc, util
from decalib.utils.renderer import SRenderY
from decalib.utils.rotation_converter import batch_euler2axis


class DECADecoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)
        self.to(device)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)

    def _create_model(self, model_cfg):
        # decoders
        self.flame = FLAME(model_cfg)
        self.flametex = FLAMETex(model_cfg)

    def visofp(self, normals):
        """visibility of keypoints, based on the normal direction"""
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).float()
        return vis68

    def decode(self, codedict, images, rendering=True, iddict=None, vis_lmk=True, return_vis=True):
        bsz = images.shape[0]

        # decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"], expression_params=codedict["exp"], pose_params=codedict["pose"]
        )
        if self.cfg.model.use_tex:
            if "static_albedo" in codedict:
                albedo = codedict["static_albedo"]
                lights = None
            else:
                albedo = self.flametex(codedict["tex"])
                lights = codedict["light"]
        else:
            albedo = torch.zeros([bsz, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        def batch_orth_proj(X, camera, transl):
            """orthgraphic projection
            X:  3d vertices, [bz, n_point, 3]
            camera: scale and translation, [bz, 3], [scale, tx, ty]
            """
            camera = camera.clone().view(-1, 1, camera.shape[-1])
            transl = transl.clone().view(-1, 1, transl.shape[-1])
            X_trans = X[:, :, :2] + transl[:, :, :2]
            X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
            Xn = camera[:, :, :1] * X_trans
            return Xn

        # projection
        landmarks2d = batch_orth_proj(landmarks2d, codedict["cam"], codedict["transl"])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        # landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = batch_orth_proj(landmarks3d, codedict["cam"], codedict["transl"])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        # landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = batch_orth_proj(verts, codedict["cam"], codedict["transl"])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        opdict = {
            "verts": verts,
            "trans_verts": trans_verts,
            "landmarks2d": landmarks2d,
            "landmarks3d": landmarks3d,
            "landmarks3d_world": landmarks3d_world,
        }

        # rendering
        if rendering:
            ops = self.render(verts, trans_verts, albedo, lights)
            # output
            opdict["grid"] = ops["grid"]
            opdict["rendered_images"] = ops["images"]
            opdict["alpha_images"] = ops["alpha_images"]
            opdict["normal_images"] = ops["normal_images"]

        if self.cfg.model.use_tex:
            opdict["albedo"] = albedo
            opdict["uv_face_eye_mask"] = self.uv_face_eye_mask

        if vis_lmk:
            landmarks3d_vis = self.visofp(ops["transformed_normals"])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict["landmarks3d"] = landmarks3d

        if not return_vis:
            return opdict, {}

        # render shape
        shape_images = self.render.render_shape(verts, trans_verts)
        # extract texture
        uv_pverts = self.render.world2uv(trans_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode="bilinear", align_corners=False)
        uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
            torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7
        )

        opdict["uv_texture_gt"] = uv_texture_gt
        visdict = {
            "inputs": images,
            "landmarks2d": util.tensor_vis_landmarks(images, landmarks2d),
            "shape_images": shape_images,
        }
        if self.cfg.model.use_tex:
            visdict["rendered_images"] = ops["images"]
        return opdict, visdict

    def save_obj(self, filename, opdict, i):
        """
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        vertices = opdict["verts"][i].detach().cpu().numpy()
        faces = self.render.faces[0].detach().cpu().numpy()
        texture = util.tensor2image(opdict["uv_texture_gt"][i])
        uvcoords = self.render.raw_uvcoords[0].detach().cpu().numpy()
        uvfaces = self.render.uvfaces[0].detach().cpu().numpy()
        # save coarse mesh, with texture and normal map
        # normal_map = util.tensor2image(opdict["uv_detail_normals"][i] * 0.5 + 0.5)
        util.write_obj(filename, vertices, faces, texture=texture, uvcoords=uvcoords, uvfaces=uvfaces)
        util.write_obj(filename.replace(".obj", "_nontex.obj"), vertices, faces, inverse_face_order=True)


class Solver(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        if self.cfg.loss.id > 0:
            self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)

    def init_parameters(self, bsz):
        codedict = {}
        for key in self.cfg.model.all_param_list:
            n = self.cfg.model.get("n_" + key)
            if key in self.cfg.model.shared_param_list:
                p = torch.zeros((1, n), device=self.device, dtype=torch.float32)
            else:
                p = torch.zeros((bsz, n), device=self.device, dtype=torch.float32)
            p.data.normal_(0, 0.01)
            if key == "cam":
                p[:, 0] = self.cfg.model.init_cam
            if key == "light":
                p = p.view(-1, n // 3, 3)
            # insert
            if key in self.cfg.model.param_list:
                p = torch.nn.Parameter(p)
            codedict[key] = p
        return codedict

    def get_loss(self, batch, opdict, codedict):
        images = batch["image"]
        lmk = batch["landmark"]
        masks = batch["mask"]  # TODO
        if lmk.shape[-1] == 2:
            lmk = torch.nn.functional.pad(lmk, (0, 1), "constant", 1.0)

        bsz = images.shape[0]

        # rendering
        if self.cfg.loss.photo > 0.0:
            # mask
            mask_face_eye = F.grid_sample(
                opdict["uv_face_eye_mask"].expand(bsz, -1, -1, -1), opdict["grid"].detach(), align_corners=False
            )
            mask_face_eye = (mask_face_eye > 0.5).float()
            # images
            predicted_images = opdict["rendered_images"] * mask_face_eye * opdict["alpha_images"]
            opdict["mask_face_eye"] = mask_face_eye
            opdict["predicted_images"] = predicted_images

        losses = {}

        # base shape
        predicted_landmarks = opdict["landmarks2d"]
        # print(predicted_landmarks)
        # print(lmk)
        if self.cfg.loss.useWlmk:
            losses["landmark"] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
        else:
            losses["landmark"] = lossfunc.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
        if self.cfg.loss.eyed > 0.0:
            losses["eye_distance"] = lossfunc.eyed_loss(predicted_landmarks, lmk) * self.cfg.loss.eyed
        if self.cfg.loss.lipd > 0.0:
            losses["lip_distance"] = lossfunc.lipd_loss(predicted_landmarks, lmk) * self.cfg.loss.lipd

        if self.cfg.loss.photo > 0.0:
            if masks is not None:
                masks = masks[:, None, :, :]
            else:
                masks = mask_face_eye * opdict["alpha_images"]
            losses["photometric_texture"] = (masks * (predicted_images - images).abs()).mean() * self.cfg.loss.photo

            mask_id = mask_face_eye * opdict["alpha_images"]
            overlay = predicted_images * mask_id + images * (1 - mask_id)
            opdict["overlay"] = overlay

        if self.cfg.loss.id > 0.0:
            # shading_images = self.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
            # albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
            # overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
            # overlay = images * mask_face_eye * opdict["alpha_images"]
            losses["identity"] = self.id_loss(overlay, images) * self.cfg.loss.id

        losses["shape_reg"] = (torch.sum(codedict["shape"] ** 2) / 2) * self.cfg.loss.reg_shape
        losses["expression_reg"] = (torch.sum(codedict["exp"] ** 2) / 2) * self.cfg.loss.reg_exp
        losses["tex_reg"] = (torch.sum(codedict["tex"] ** 2) / 2) * self.cfg.loss.reg_tex
        losses["light_reg"] = (
            (torch.mean(codedict["light"], dim=2)[:, :, None] - codedict["light"]) ** 2
        ).mean() * self.cfg.loss.reg_light
        if self.cfg.model.jaw_type == "euler":
            # import ipdb; ipdb.set_trace()
            # reg on jaw pose
            losses["reg_jawpose_roll"] = (torch.sum(codedict["euler_jaw_pose"][:, -1] ** 2) / 2) * 100.0
            losses["reg_jawpose_close"] = (torch.sum(F.relu(-codedict["euler_jaw_pose"][:, 0]) ** 2) / 2) * 10.0

        # ########################################################
        all_loss = 0.0
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses["all_loss"] = all_loss
        return losses, opdict

    def visualize(self, visdict, size=None):
        grids = {}
        if size is None:
            size = self.image_size
        bsz = 1
        for key in visdict:
            bsz = visdict[key].shape[0]
        rows = []
        for i in range(bsz):
            for key in visdict:
                grids[key] = (
                    torchvision.utils.make_grid(F.interpolate(visdict[key][i : i + 1], [size, size])).detach().cpu()
                )
            grid = torch.cat(list(grids.values()), 2)
            grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
            rows.append(grid_image)
        return np.concatenate(rows, axis=0)
