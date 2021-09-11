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
import sys
import toml
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from decalib.utils.renderer import SRenderY
from decalib.models.encoders import ResnetEncoder
from decalib.models.FLAME import FLAME, FLAMETex
from decalib.models.decoders import Generator
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis
from decalib.datasets import datasets
from decalib.utils.config import cfg
from decalib.utils import lossfunc

torch.backends.cudnn.benchmark = True
A = 256

cfg.dataset.image_size = A
# cfg.loss.photo = 0
# cfg.loss.id = 0
# cfg.loss.useWlmk = False
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl') 
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'transl', 'light']
cfg.model.n_shape = 300
cfg.model.n_tex = 50
cfg.model.n_exp = 100
cfg.model.n_cam = 1
cfg.model.n_transl = 2
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.init_cam = 7.0
cfg.loss.useWlmk = True
cfg.loss.lmk = 5.0
cfg.loss.eyed = 0  # 1.0
cfg.loss.lipd = 0  # 0.5
cfg.loss.photo = 1.0
cfg.loss.useSeg = True
cfg.loss.id = 0.01
cfg.loss.id_shape_only = True
cfg.loss.reg_shape = 1e-04
cfg.loss.reg_exp = 1e-04
cfg.loss.reg_tex = 0  # 1e-05
cfg.loss.reg_light = 1.
cfg.loss.reg_jaw_pose = 0.  # 1.
cfg.loss.use_gender_prior = False
cfg.loss.shape_consistency = True


class DECADecoder(nn.Module):
    def __init__(self, config=None, device="cuda"):
        super().__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

        if self.cfg.loss.id > 0:
            self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).to(
            self.device
        )
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32) / 255.0
        mean_texture = torch.from_numpy(mean_texture.transpose(2, 0, 1))[None, :, :, :].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding="latin1").item()

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = (
            model_cfg.n_shape
            + model_cfg.n_tex
            + model_cfg.n_exp
            + model_cfg.n_pose
            + model_cfg.n_cam
            + model_cfg.n_light
        )
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [
            model_cfg.n_shape,
            model_cfg.n_tex,
            model_cfg.n_exp,
            model_cfg.n_pose,
            model_cfg.n_cam,
            model_cfg.n_light,
        ]
        self.param_dict = OrderedDict({i: model_cfg.get("n_" + i) for i in model_cfg.param_list})

        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
    
    def init_parameters(self, bsz):
        codedict = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'cam', 'transl', 'light']:
            n = self.cfg.model.get("n_" + key)
            if key in ['shape', 'tex', 'cam', 'light']:
                p = torch.zeros((1, n), device=self.device, dtype=torch.float32)
            else:
                p = torch.zeros((bsz, n), device=self.device, dtype=torch.float32)
            p.data.normal_(0, 0.01)
            if key == 'cam':
                p[:, 0] = self.cfg.model.init_cam
            if key == 'light':
                p = p.view(-1, n//3, 3)
            # insert
            if key in self.cfg.model.param_list:
                p = torch.nn.Parameter(p)
            codedict[key] = p
        return codedict

    def decompose_code(self, code, num_dict):
        """Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        """
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == "light":
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def visofp(self, normals):
        """visibility of keypoints, based on the normal direction"""
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).float()
        return vis68

    def decode(self, codedict, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True):
        images = codedict["images"]
        batch_size = images.shape[0]

        # decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"], expression_params=codedict["exp"], pose_params=codedict["pose"]
        )
        if self.cfg.model.use_tex:
            if 'static_albedo' in codedict:
                albedo = codedict['static_albedo']
            else:
                albedo = self.flametex(codedict["tex"])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        def batch_orth_proj(X, camera, transl):
            ''' orthgraphic projection
                X:  3d vertices, [bz, n_point, 3]
                camera: scale and translation, [bz, 3], [scale, tx, ty]
            '''
            camera = camera.clone().view(-1, 1, 1)
            transl = transl.clone().view(-1, 1, 2)
            X_trans = X[:, :, :2] + transl
            X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
            Xn = camera * X_trans
            return Xn

        # projection
        landmarks2d = batch_orth_proj(landmarks2d, codedict["cam"], codedict['transl'])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        # landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = batch_orth_proj(landmarks3d, codedict["cam"], codedict['transl'])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        # landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = batch_orth_proj(verts, codedict["cam"], codedict['transl'])
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
            ops = self.render(verts, trans_verts, albedo, codedict["light"])
            # output
            opdict["grid"] = ops["grid"]
            opdict["rendered_images"] = ops["images"]
            opdict["alpha_images"] = ops["alpha_images"]
            opdict["normal_images"] = ops["normal_images"]

        if self.cfg.model.use_tex:
            opdict["albedo"] = albedo

        if vis_lmk:
            landmarks3d_vis = self.visofp(ops["transformed_normals"])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict["landmarks3d"] = landmarks3d

        if return_vis:
            ##render shape
            shape_images = self.render.render_shape(verts, trans_verts)
            # extract texture
            ##TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode="bilinear", align_corners=False)
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7
            )

            opdict["uv_texture_gt"] = uv_texture_gt
            visdict = {
                "inputs": images,
                # "landmarks2d_real": util.tensor_vis_landmarks(images, codedict['lmk']),
                "landmarks2d": util.tensor_vis_landmarks(images, landmarks2d),
                # "landmarks3d": util.tensor_vis_landmarks(images, landmarks3d),
                "shape_images": shape_images,
            }
            if self.cfg.model.use_tex:
                visdict["rendered_images"] = ops["images"]
            return opdict, visdict

        else:
            return opdict

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
                grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key][i:i+1], [size, size])).detach().cpu()
            grid = torch.cat(list(grids.values()), 2)
            grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
            rows.append(grid_image)
        return np.concatenate(rows, axis=0)

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
        util.write_obj(
            filename, vertices, faces, texture=texture, uvcoords=uvcoords, uvfaces=uvfaces, # normal_map=normal_map
        )
        util.write_obj(
            filename.replace('.obj', '_nontex.obj'), vertices, faces, inverse_face_order=True
        )
        # # upsample mesh, save detailed mesh
        # texture = texture[:, :, [2, 1, 0]]
        # normals = opdict["normals"][i].detach().cpu().numpy()
        # displacement_map = opdict["displacement_map"][i].detach().cpu().numpy().squeeze()
        # dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
        #     vertices, normals, faces, displacement_map, texture, self.dense_template
        # )
        # util.write_obj(
        #     filename.replace(".obj", "_detail.obj"),
        #     dense_vertices,
        #     dense_faces,
        #     # colors = dense_colors,
        #     inverse_face_order=True,
        # )

    def get_loss(self, batch, opdict):
        images = batch['image']
        lmk = batch['landmark']
        masks = batch['mask']  # TODO
        opdict['images'] = batch['image']
        opdict['lmk'] = batch['landmark']

        bsz = images.shape[0]

        # rendering
        if self.cfg.loss.photo > 0.:
            # mask
            mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(bsz,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
            # images
            predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
            opdict['mask_face_eye'] = mask_face_eye
            opdict['predicted_images'] = predicted_images

        losses = {}
        
        # base shape
        predicted_landmarks = opdict['landmarks2d']
        # print(predicted_landmarks)
        # print(lmk)
        if self.cfg.loss.useWlmk:
            losses['landmark'] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
        else:    
            losses['landmark'] = lossfunc.landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
        if self.cfg.loss.eyed > 0.:
            losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk)*self.cfg.loss.eyed
        if self.cfg.loss.lipd > 0.:
            losses['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks, lmk)*self.cfg.loss.lipd
        
        if self.cfg.loss.photo > 0.:
            if masks is not None:
                masks = masks[:,None,:,:]
            else:
                masks = mask_face_eye*opdict['alpha_images']
            losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()*self.cfg.loss.photo

            mask_id = mask_face_eye * opdict["alpha_images"]
            overlay = predicted_images * mask_id + images * (1-mask_id)
            opdict['overlay'] = overlay

        if self.cfg.loss.id > 0.:
            # shading_images = self.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
            # albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
            # overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
            # overlay = images * mask_face_eye * opdict["alpha_images"]
            losses['identity'] = self.id_loss(overlay, images) * self.cfg.loss.id
        
        losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.loss.reg_shape
        losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
        losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
        losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
        if self.cfg.model.jaw_type == 'euler':
            # import ipdb; ipdb.set_trace()
            # reg on jaw pose
            losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:,-1]**2)/2)*100.
            losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:,0])**2)/2)*10.

        # ########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict


def read_image(fpath, A):
    img = cv2.imread(fpath)[..., [2, 1, 0]]
    mask = cv2.imread(os.path.splitext(fpath)[0] + "_mask.png")[..., 0]
    with open(os.path.splitext(fpath)[0] + "_lmks.toml") as fp:
        lmks = toml.load(fp)['points']
        lmks = np.asarray(lmks, dtype=np.float32)
        lmks[:, 0] /= img.shape[1]
        lmks[:, 1] /= img.shape[0]
        lmks = lmks * 2 - 1
    img = (cv2.resize(img, (A, A)) / 255.0).astype(np.float32)
    mask = (cv2.resize(mask, (A, A)) / 255.0).astype(np.float32)
    return img, mask, lmks


bsz = 4
lr = 1e-2
decoder = DECADecoder()

# batch
img_list, mask_list, lmk_list = [], [], []
for i in range(bsz):
    img, mask, lmks = read_image(f"TestSamples/obama/image_{i+1:02d}.png", A=A)
    img_list.append(img)
    mask_list.append(mask)
    lmk_list.append(lmks)

image = torch.tensor(img_list, device=decoder.device).permute(0, 3, 1, 2)
landmark = torch.tensor(lmk_list, device=decoder.device)
mask = torch.tensor(mask_list, device=decoder.device)


def optimize(param_dict, optim, n_iters):
    codedict = {k: v for k, v in param_dict.items()}
    for k in codedict:
        if codedict[k].shape[0] == 1:
            codedict[k] = codedict[k].expand(bsz, *([-1]*(codedict[k].ndim-1)))

    for i_iter in tqdm(range(n_iters)):
        batch = dict()
        batch['image'] = image
        batch['mask'] = mask
        batch['landmark'] = torch.nn.functional.pad(landmark, (0, 1), "constant", 1.0)

        # for k, v in codedict.items():
        #     print(k, v.shape, torch.any(v.isnan()), v.abs().min(), v.abs().max())
        print('cam', codedict['cam'].detach().cpu().numpy()[:, 0])
        codedict['images'] = batch['image']
        codedict['lmk'] = batch['landmark']
        opdict, visdict = decoder.decode(codedict)
        # loss
        losses, opdict = decoder.get_loss(batch, opdict)

        # optim
        optim.zero_grad()
        losses['all_loss'].backward()
        optim.step()
        factor = np.clip(np.exp(-(i_iter-100)/100), 0.1, 1.0)
        for group in optim.param_groups:
            group['lr'] = lr * factor

        # vis
        visdict['predicted_images'] = opdict['predicted_images'] * batch['mask'][:, None, :, :]
        if 'overlay' in opdict:
            visdict['overlay'] = opdict['overlay']
        if 'albedo' in opdict:
            visdict['albedo'] = opdict['albedo']
        canvas = decoder.visualize(visdict)
        cv2.imshow('canvas', cv2.resize(canvas, None, fx=0.75, fy=0.75))
        cv2.waitKey(1)

    cv2.imwrite("TestSamples/obama/results/canvas.png", canvas)
    for i in range(bsz):
        decoder.save_obj(f"TestSamples/obama/results/image_{i:02d}/recons.obj", opdict, i=i)

    return opdict


decoder.cfg.loss.id = 0

codedict = decoder.init_parameters(bsz)
optim = torch.optim.Adam([v for k, v in codedict.items()], lr=lr)
opdict = optimize(codedict, optim, 500)

codedict['static_albedo'] = torch.nn.Parameter(opdict['albedo'][:1].detach().clone())
optim = torch.optim.Adam([codedict['static_albedo'], codedict['light']], lr=lr)
optimize(codedict, optim, 500)
