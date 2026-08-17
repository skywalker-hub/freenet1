"""HyperImageNet 竞赛子集的 SimpleCV 训练入口。

与官方 train.py 并列，不覆盖它。两处实质差别：

1. 评估跨全部测试瓦片累积一个混淆矩阵再算指标。官方 train.py 的 fcn_evaluate_fn
   把 oa/aa/kappa 的计算写在了 for 循环体内，循环结束后只剩最后一张图的值——
   Pavia 只有 1 张测试图所以从没暴露，换成上百张瓦片就是错的。
2. 主指标换成 macro-F1，并且区分 "验证集里实际出现的类别" 和 "全部类别" 两个口径。

用法：
    bash scripts/freenet_1_0_hin.sh
或者：
    python train_hin.py --config_path freenet.freenet_1_0_hin --model_dir ./log/hin
"""

import time

import torch
from simplecv import dp_train as train
from simplecv.util import registry
from simplecv.util.logger import eval_progress, speed

from data import hin_tiles  # noqa: F401  导入即注册 HINTileLoader
from hin.labels import IGNORE_INDEX, class_name, target_index_to_submission_id
from hin.metrics import ConfusionMatrix, format_per_class
from module import freenet_hin  # noqa: F401  导入即注册 FreeNetHIN

# 官方只注册了 sgd / adam
if 'adamw' not in registry.OPT:
    registry.OPT.register('adamw', torch.optim.AdamW)


def hin_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return

    model = self._model.module if hasattr(self._model, 'module') else self._model
    n_cls = model.config.num_classes
    mode = 'open' if n_cls == 32 else 'closed'
    device = next(model.parameters()).device

    self._model.eval()
    cm = ConfusionMatrix(n_cls, device=device)
    total_time = 0.0
    with torch.no_grad():
        for idx, (image, target, _) in enumerate(test_dataloader):
            start = time.time()
            prob = self._model(image.to(device, non_blocking=True))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - start

            cm.update(target.to(device, non_blocking=True), prob.argmax(dim=1), IGNORE_INDEX)
            if (idx + 1) % 20 == 0:
                eval_progress(self._logger, idx + 1, len(test_dataloader))
    self._model.train()

    speed(self._logger, round(total_time / max(len(test_dataloader), 1), 3), 'im (avg)')

    res = cm.compute()
    names = [class_name(mode, i) for i in range(n_cls)]
    self._logger.info('\n' + format_per_class(res, names, target_index_to_submission_id(mode)))

    metric_dict = {
        'OA': float(res['oa']),
        'macro_F1_all': float(res['macro_f1_all']),
        'macro_F1_present': float(res['macro_f1_present']),
        'num_present_classes': float(res['num_present_classes']),
        'F1': res['f1'],
    }
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)


def register_evaluate_fn(launcher):
    launcher.override_evaluate(hin_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts)
