import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from decalib.utils import util

from .solver import DECADecoder, Solver
from .utils import read_image


def solve_identity(source_dir, export_dir, cfg, device='cuda:0'):
    bsz = 8
    lr = cfg.train.lr
    A = cfg.dataset.image_size

    # build object
    decoder = DECADecoder(cfg, device)
    solver = Solver(cfg, device)

    def optimize(batch, param_dict, optim, n_iters):
        codedict = {k: v for k, v in param_dict.items()}
        for k in codedict:
            if codedict[k].shape[0] == 1:
                codedict[k] = codedict[k].expand(bsz, *([-1] * (codedict[k].ndim - 1)))

        for i_iter in tqdm(range(n_iters)):
            # decode codes
            opdict, visdict = decoder.decode(codedict, batch["image"], batch['landmark'])
            # calculate loss and update outputs
            lossdict, opdict = solver.get_loss(batch, opdict, codedict)
            loss = lossdict['all_loss']

            # optim step
            optim.zero_grad()
            loss.backward()
            optim.step()
            # update learning rate
            factor = np.clip(np.exp(-(i_iter - 100) / 100), 0.1, 1.0)
            for group in optim.param_groups:
                group["lr"] = lr * factor

            # vis
            # if "predicted_images" in opdict:
            #     visdict["predicted_images"] = opdict["predicted_images"]  # * batch['mask'][:, None, :, :]
            if "overlay" in opdict:
                visdict["overlay"] = opdict["overlay"]
            if "albedo" in opdict:
                visdict["albedo"] = opdict["albedo"]
            
            if i_iter % 1 == 0:
                canvas = solver.visualize(visdict)
                cv2.imshow("canvas", cv2.resize(canvas, None, fx=0.5, fy=0.5))
                cv2.waitKey(1)

        return opdict, canvas
    
    # read data
    img_list, mask_list, lmk_list = [], [], []
    for i in range(bsz):
        img, mask, lmks = read_image(f"TestSamples/obama/image_{i+1:02d}.png", A=A)
        img_list.append(img)
        mask_list.append(mask)
        lmk_list.append(lmks)
    image = torch.tensor(img_list, device=device).permute(0, 3, 1, 2)
    landmark = torch.tensor(lmk_list, device=device)
    mask = torch.tensor(mask_list, device=device)

    # * ------------------------------------------ phase 1: solve geometry ----------------------------------------- * #
    # update loss configuration
    solver.cfg.loss.id = 0
    # get batch
    batch = dict(image=image, mask=mask, landmark=landmark)
    # init parameters and optim
    codedict = solver.init_parameters(bsz)
    optim = torch.optim.Adam([v for k, v in codedict.items()], lr=lr)
    # optimize
    opdict, canvas = optimize(batch, codedict, optim, 500)

    # * --------------------------------------- phase 2: solve static albedo --------------------------------------- * #
    # update loss configuration
    solver.cfg.loss.id = 0
    # get batch
    batch = dict(image=image, mask=None, landmark=landmark)
    # init parameter and optim
    codedict["static_albedo"] = torch.nn.Parameter(opdict["albedo"][:1].detach().clone())
    raw = opdict["albedo"][0].permute(1, 2, 0).detach().cpu().numpy()
    raw = cv2.resize(raw, (128, 128))
    codedict["static_albedo"] = torch.nn.Parameter(torch.tensor(raw, device=device).permute(2, 0, 1).unsqueeze(0))
    optim = torch.optim.Adam([codedict["static_albedo"], codedict["light"]], lr=lr)
    # optimize
    opdict, canvas = optimize(batch, codedict, optim, 500)

    # * ---------------------------------------------- export results ---------------------------------------------- * #
    odir = os.path.join(export_dir, "frames")
    os.makedirs(odir, exist_ok=True)
    cv2.imwrite(f"{export_dir}/results.png", canvas)
    for i in range(bsz):
        decoder.save_obj(f"{export_dir}/image_{i:02d}/recons.obj", opdict, i=i)

    # static texture
    odir = os.path.join(export_dir, "identity")
    os.makedirs(odir, exist_ok=True)
    albedo = util.tensor2image(codedict['static_albedo'][0])
    cv2.imwrite(f"{odir}/albedo.png", albedo)

    # shape
    shape = codedict["shape"][0].detach().cpu().numpy().astype(np.float32)
    np.save(f"{odir}/shape.npy", shape)

    # template
    verts, _, _ = decoder.flame(
        shape_params=codedict["shape"][:1],
        expression_params=torch.zeros_like(codedict["exp"][:1]),
        pose_params=torch.zeros_like(codedict["pose"][:1]),
    )
    verts = verts[0].detach().cpu().numpy()
    faces = decoder.render.faces[0].detach().cpu().numpy()
    uvcoords = decoder.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = decoder.render.uvfaces[0].detach().cpu().numpy()
    util.write_obj(f"{odir}/template_tex.obj", verts, faces, texture=albedo, uvcoords=uvcoords, uvfaces=uvfaces)
    util.write_obj(f"{odir}/template.obj", verts, faces, inverse_face_order=True)
