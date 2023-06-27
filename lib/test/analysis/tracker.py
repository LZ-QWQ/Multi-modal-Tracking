from lib.test.evaluation.environment import env_settings


def trackerlist(
    name: str,
    parameter_name: str,  # 兼容eval相关的代码..., 其实置空也行
    run_ids=None,
    display_name: str = None,
    dataset_name: str = "",
    save_path="",  # save_path/dataset_name/{}.txt
    tracker_type: str = "RGBT",
):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]

    if tracker_type == "RGBT":
        return [RGBT_Tracker(name, parameter_name, run_id, display_name, dataset_name, save_path) for run_id in run_ids]
    else:
        raise ValueError


# 相比test时用的更简洁....定义稍微不太一样
class RGBT_Tracker:
    def __init__(
        self,
        name: str,
        parameter_name: str,  # 兼容eval相关的代码..., 其实置空也行
        run_id: int = None,
        display_name: str = None,
        dataset_name: str = "",
        save_path="",  # save_path/dataset_name/{}.txt
    ):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.dataset_name = dataset_name
        self.display_name = display_name

        # 这两玩意好像是能用来合并多次测试的
        self.parameter_name = parameter_name
        self.run_id = run_id

        env = env_settings()
        if self.run_id is None:
            self.results_dir = "{}/{}/{}".format(env.results_path, self.name, save_path)
        else:

            self.results_dir = "{}/{}/{}_{}".format(env.results_path, self.name, save_path, self.run_id)
