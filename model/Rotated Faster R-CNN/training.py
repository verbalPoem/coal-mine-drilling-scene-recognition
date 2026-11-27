"""Single-device training and validation hooks."""
import math

from mmcv.runner import EpochBasedRunner
from mmdet.apis import single_gpu_test
from mmdet.core import EvalHook


class TrainingRunner(EpochBasedRunner):
    def load_checkpoint(self, filename, map_location="cpu", **kwargs):
        checkpoint = super().load_checkpoint(filename, map_location=map_location, strict=True)
        if "optimizer" not in checkpoint:
            raise ValueError("This checkpoint has no optimizer state. Use --weights to initialize a new run.")
        saved_names = checkpoint.get("meta", {}).get("CLASSES")
        if saved_names and tuple(saved_names) != tuple(self.model.module.CLASSES):
            raise ValueError("Dataset class order differs from the checkpoint.")
        return checkpoint


class BestValidationHook(EvalHook):
    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        score = self.evaluate(runner, results)
        if self.save_best and score is not None:
            if not math.isfinite(score):
                raise ValueError("Validation returned a non-finite checkpoint score.")
            # Keep prior improvements, including checkpoints from an earlier run directory.
            self.best_ckpt_path = None
            self._save_ckpt(runner, score)
