"""HyperImageNet 竞赛子集的 SimpleCV 训练入口。

与官方 train.py 并列，不覆盖它。三处实质差别：

1. 评估跨全部测试瓦片累积一个混淆矩阵再算指标。官方 train.py 的 fcn_evaluate_fn
   把 oa/aa/kappa 的计算写在了 for 循环体内，循环结束后只剩最后一张图的值——
   Pavia 只有 1 张测试图所以从没暴露，换成上百张瓦片就是错的。
2. 主指标换成 macro-F1，并且区分 "验证集里实际出现的类别" 和 "全部类别" 两个口径。
3. 支持多个验证集。官方一次只评一个 data.test，而这里线上测试集是混合的：
   179 张测试瓦片里 41 张与训练瓦片空间重叠，另有 18 条航线训练完全没见过。
   同域和留出航线两个分数的差值就是域偏移的代价，只看一个会误判。

用法：
    bash scripts/freenet_1_0_hin.sh
或者：
    python train_hin.py --config_path freenet.freenet_1_0_hin --model_dir ./log/hin
"""

import time

import torch
from simplecv import dp_train as train
from simplecv.core.config import AttrDict
from simplecv.data.data_loader import make_dataloader
from simplecv.util import config as config_util
from simplecv.util import registry
from simplecv.util.logger import eval_progress, speed

from data import hin_tiles  # noqa: F401  导入即注册 HINTileLoader
from hin.labels import IGNORE_INDEX, class_name, target_index_to_submission_id
from hin.metrics import ConfusionMatrix, format_per_class
from module import freenet_hin  # noqa: F401  导入即注册 FreeNetHIN

# 官方只注册了 sgd / adam
if 'adamw' not in registry.OPT:
    registry.OPT.register('adamw', torch.optim.AdamW)

# data 下除 train/test 外，还会被当作验证集加载的键
EXTRA_VAL_KEYS = ('val_line',)

# --val none 时追加的配置覆盖：训练集换回全部 383 张瓦片，并关掉所有评估。
# 划分文件只是为了留出验证数据，不验证了就没有理由把那 102 张排除在训练之外。
FULL_DATA_OPTS = ('data.train.params.split_file', 'None',
                  'train.eval_per_epoch', 'False',
                  'train.eval_after_train', 'False')


def model_of(launcher):
    model = launcher._model.module if hasattr(launcher._model, 'module') else launcher._model
    n_cls = model.config.num_classes
    return model, n_cls, 'open' if n_cls == 32 else 'closed'


def run_inference(self, dataloader):
    model, n_cls, _ = model_of(self)
    device = next(model.parameters()).device

    self._model.eval()
    cm = ConfusionMatrix(n_cls, device=device)
    total_time = 0.0
    with torch.no_grad():
        for idx, (image, target, _) in enumerate(dataloader):
            start = time.time()
            prob = self._model(image.to(device, non_blocking=True))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - start

            cm.update(target.to(device, non_blocking=True), prob.argmax(dim=1), IGNORE_INDEX)
            if (idx + 1) % 20 == 0:
                eval_progress(self._logger, idx + 1, len(dataloader))
    self._model.train()

    speed(self._logger, round(total_time / max(len(dataloader), 1), 3), 'im (avg)')
    return cm


def log_metrics(self, cm, tag):
    _, n_cls, mode = model_of(self)
    res = cm.compute()
    names = [class_name(mode, i) for i in range(n_cls)]
    self._logger.info(f'[{tag}]\n'
                      + format_per_class(res, names, target_index_to_submission_id(mode)))

    self._logger.eval_log(metric_dict={
        f'{tag}/OA': float(res['oa']),
        f'{tag}/macro_F1_all': float(res['macro_f1_all']),
        f'{tag}/macro_F1_present': float(res['macro_f1_present']),
        f'{tag}/num_present_classes': float(res['num_present_classes']),
        f'{tag}/F1': res['f1'],
    }, step=self.checkpoint.global_step)
    return res


def log_domain_gap(self, results, min_support=1000):
    """在两边共有的类别上重算 macro-F1，这样域偏移的差值才有意义。

    各自的 macro_F1_present 不可比：分母是各自出现过的类别集，indomain 有 27 类
    而 val_line 只有 18 类，类别集不同就不是同一把尺子。取交集重算才是同一把。
    support 太小的类别也排除掉——val_line 里 Barren 只有 144 个像元，它的 F1
    是采样噪声，不是域偏移。
    """
    common = None
    for _, res in results:
        enough = res['support'] >= min_support
        common = enough if common is None else (common & enough)
    if common is None or not common.any():
        return

    self._logger.info(f'[common] {int(common.sum())} 个两边 support 都 >= {min_support} 的类别')
    self._logger.eval_log(
        metric_dict={f'common/{tag}_macro_F1': float(res['f1'][common].mean())
                     for tag, res in results},
        step=self.checkpoint.global_step)


def make_evaluate_fn(extra_loaders):
    def hin_evaluate_fn(self, test_dataloader, config):
        if self.checkpoint.global_step < 0:
            return
        mats = [('indomain', run_inference(self, test_dataloader))]
        for tag, loader in extra_loaders:
            mats.append((tag, run_inference(self, loader)))

        results = [(tag, log_metrics(self, cm, tag)) for tag, cm in mats]
        if len(mats) > 1:
            # combined 是最接近线上分数的本地读数：单个验证集缺 5-14 个类，
            # 那些类在 macro_F1_all 里按 0 计入分母，分数被结构性压低；
            # 合起来覆盖 30/32 类，上限从 0.56 抬到 0.94，才好跟排行榜比。
            log_metrics(self, ConfusionMatrix.merge([cm for _, cm in mats]), 'combined')
            log_domain_gap(self, results)

    return hin_evaluate_fn


def build_extra_loaders(config_path, opts):
    """dp_train.run 只认 data.train 和 data.test，第二个验证集得自己建。

    这里重复一遍 dp_train 的配置解析（import + 命令行覆盖），才能保证脚本里
    `data.val_line.params.limit 4` 这类覆盖对两个验证集同样生效。
    """
    cfg = AttrDict.from_dict(config_util.import_config(config_path))
    if opts:
        cfg.update_from_list(opts)
    return [(key, make_dataloader(cfg['data'][key]))
            for key in EXTRA_VAL_KEYS if key in cfg['data']]


if __name__ == '__main__':
    train.parser.add_argument(
        '--val', default='both', choices=('both', 'single', 'none'),
        help='both=同域与留出航线两个验证集；single=只评 data.test（自检用）；'
             'none=完全不验证，训练集换成全部 383 张瓦片')
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    opts = list(args.opts or [])
    if args.val == 'none':
        opts += list(FULL_DATA_OPTS)
    extra = build_extra_loaders(args.config_path, opts) if args.val == 'both' else []
    evaluate_fn = make_evaluate_fn(extra)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[lambda tl: tl.override_evaluate(evaluate_fn)],
              opts=opts)
