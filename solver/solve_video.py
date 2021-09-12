import os
import re
import cv2
import toml
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from decalib.utils import lossfunc, util

from .solver import DECADecoder, Solver
from .utils import read_image


def read_data(image_dir, lmks_toml):
    image_list = glob(os.path.join(image_dir, "*.png"))
    image_list = sorted(
        [x for x in image_list if re.match(r"\d+.png", os.path.basename(x))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    with open(lmks_toml, "r") as fp:
        data = toml.load(fp)
        lmks_list = [x['points'] for x in data['frames']]
        lmks_list = np.asarray(lmks_list, dtype=np.float32)
        lmks_list[..., 0] /= float(data['resolution'][0])
        lmks_list[..., 1] /= float(data['resolution'][0])
        lmks_list = lmks_list * 2 - 1
    return image_list, lmks_list


def solve_video(image_dir, lmks_toml, cfg, device):
    lr = cfg.train.lr
    A = cfg.dataset.image_size

    # build object
    cfg.model.param_list = ["exp", "pose", "cam", "transl"]
    cfg.loss.id = 0
    cfg.loss.lmks = 4.0
    cfg.loss.reg_shape = 0
    cfg.loss.reg_exp = 1e-6

    decoder = DECADecoder(cfg, device)
    solver = Solver(cfg, device)

    export_dir = "TestSamples/obama/results/trn-000"
    os.makedirs(os.path.join(export_dir, "frames"), exist_ok=True)

    shape = np.load("TestSamples/obama/results/identity/shape.npy")
    albedo = (cv2.imread("TestSamples/obama/results/identity/albedo.png")[..., [2, 1, 0]] / 255).astype(np.float32)
    albedo = np.transpose(albedo, (2, 0, 1))
    param_dict = solver.init_parameters(1)
    param_dict['shape'].data = torch.tensor(shape[None, ...], dtype=torch.float32, device=device)
    param_dict['static_albedo'] = torch.tensor(albedo[None, ...], dtype=torch.float32, device=device)

    codedict = {k: v for k, v in param_dict.items()}
    # optim = torch.optim.Adam([
    #     {'params': [codedict[k] for k in ['exp']]},
    #     {'params': [codedict[k] for k in ['pose', 'cam', 'transl']]}
    # ], lr=lr)
    optim = torch.optim.Adam([codedict[k] for k in cfg.model.param_list], lr=lr)

    image_list, lmks_list = read_data(image_dir, lmks_toml)
    results_list = []
    for i_frame in tqdm(range(len(image_list)), desc="Frames"):
        img_fpath = image_list[i_frame]
        lmks = lmks_list[i_frame]
        img = cv2.imread(img_fpath)[..., [2, 1, 0]]
        mask = cv2.imread(os.path.splitext(img_fpath)[0] + "_mask.png")[..., 0]
        img = (cv2.resize(img, (A, A)) / 255.0).astype(np.float32)
        mask = (cv2.resize(mask, (A, A)) / 255.0).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        mask[mask > 0.95] = 1.0

        batch = dict(
            image=torch.tensor(img, device=device)[None, ...],
            mask=torch.tensor(mask, device=device)[None, ...],
            landmark=torch.tensor(lmks, device=device)[None, ...],
        )

        n_iters = 300 if i_frame == 0 else 200
        last_losses = dict(landmark=1000, photometric_texture=1000)
        delta_losses = {k: 0 for k in last_losses}
        lastcodes = {k: v.clone().detach() for k, v in codedict.items()}
        pbar_iters = tqdm(range(n_iters), desc="Iter", leave=False)
        for i_iter in pbar_iters:
            # decode codes
            opdict, visdict = decoder.decode(codedict, batch["image"])
            # calculate loss and update outputs
            lossdict, opdict = solver.get_loss(batch, opdict, codedict)
            # smooth regs
            smth_reg_exp   = (torch.sum((codedict['exp']-lastcodes['exp'])**2)/2) * 1e-4
            smth_reg_pose  = (torch.sum((codedict['pose']-lastcodes['pose'])**2)/2) * 0
            smth_reg_cam   = (torch.sum((codedict['cam']-lastcodes['cam'])**2)/2) * 1e-2
            smth_reg_trnsl = (torch.sum((codedict['transl']-lastcodes['transl'])**2)/2) * 1e-2

            # vis
            if "predicted_images" in opdict:
                visdict["predicted_images"] = opdict["predicted_images"] * batch['mask'][:, None, :, :]
            if "overlay" in opdict:
                visdict["overlay"] = opdict["overlay"]

            convergence = True
            for k in last_losses:
                delta_losses[k] = abs(float(last_losses[k] - lossdict[k]))
                last_losses[k] = float(lossdict[k])
                # print(f"{k}: {lossdict[k]:.6f}, {delta_losses[k]:.6f}")
                if k == 'landmark' and (delta_losses[k] > 6e-4 or lossdict[k] > 0.025):
                    convergence = False
                elif k == 'photometric_texture' and (delta_losses[k] > 5e-4 or lossdict[k] > 0.015):
                    convergence = False
            if convergence:
                canvas = solver.visualize(visdict)
                cv2.imshow("canvas", canvas)
                cv2.waitKey(1)
                break
            pbar_iters.set_postfix(last_losses)

            # backward
            if i_frame > 0:
                loss = lossdict['all_loss'] + smth_reg_exp + smth_reg_pose + smth_reg_cam + smth_reg_trnsl
            else:
                loss = lossdict['all_loss']
            optim.zero_grad()
            loss.backward()
            # optim step
            optim.step()
            # update learning rate
            factor = np.clip(np.exp(-(i_iter - 100) / 100), 0.1, 1.0) if i_frame == 0 else 0.1
            for group in optim.param_groups:
                group["lr"] = lr * factor

            if i_frame == 0 or i_iter + 1 == n_iters:
                canvas = solver.visualize(visdict)
                cv2.imshow("canvas", canvas)
                cv2.waitKey(1)

        # canvas = util.tensor_vis_landmarks(batch['image'], batch['landmark'])[0]
        # canvas = util.tensor2image(canvas)
        # cv2.imshow('canvas', canvas)
        # cv2.waitKey(1)

        # save results
        cv2.imwrite(os.path.join(export_dir, "frames", f"{i_frame}.png"), canvas)
        results_list.append({k: v.detach().cpu().numpy() for k, v in codedict.items()})
        if i_frame == 100:
            break
    
    for k in cfg.model.param_list:
        values = np.concatenate([x[k] for x in results_list])
        print(k, values.shape)
        np.save(os.path.join(export_dir, f"{k}.npy"), values)
