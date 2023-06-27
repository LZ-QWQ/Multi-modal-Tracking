from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.asymmetric_shared_ce.config import cfg, update_config_from_file


def parameters(training_yaml_name: str, tracking_yaml_name: str, model=None, search_area_scale=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, "experiments/asymmetric_shared_ce/%s.yaml" % training_yaml_name)
    update_config_from_file(yaml_file)  # 用training阶段的文件控制模型相关参数,
    yaml_file = os.path.join(prj_dir, "experiments/%s.yaml" % tracking_yaml_name)
    update_config_from_file(yaml_file)  # 用tracking阶段的文件来控制可能的跟踪超参数

    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    if search_area_scale is not None:
        params.search_factor = search_area_scale
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR
    # print("search_area_scale: {}".format(params.search_factor))
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if params.cfg.TEST.LOAD_FROME_TRAIN_RESULT:
        params.checkpoint = os.path.join(save_dir, model)
    else:
        params.checkpoint = os.path.join(save_dir, "models/%s" % model)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
