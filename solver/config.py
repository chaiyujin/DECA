import os


def get_default_config():
    from decalib.utils.config import cfg

    cfg.dataset.image_size = 256
    # cfg.loss.photo = 0
    # cfg.loss.id = 0
    # cfg.loss.useWlmk = False
    cfg.model.flame_model_path = os.path.join(cfg.deca_dir, "data", "generic_model.pkl")
    cfg.model.all_param_list = ["shape", "tex", "exp", "pose", "cam", "transl", "light"]
    cfg.model.param_list = ["shape", "tex", "exp", "pose", "cam", "transl", "light"]
    cfg.model.shared_param_list = ["shape", "tex", "light"]
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
    cfg.loss.reg_light = 1.0
    cfg.loss.reg_jaw_pose = 0.0  # 1.
    cfg.loss.use_gender_prior = False
    cfg.loss.shape_consistency = True
    return cfg