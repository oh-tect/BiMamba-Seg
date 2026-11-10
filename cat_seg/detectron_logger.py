import os
import sys
import threading
import torch.distributed as dist

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from detectron2.utils.events import get_event_storage


class DetectronLogger:
    """Log scalars either to TensorBoard (default) or to Detectron2's EventStorage.

    - By default `log_scalar(name, value)` writes to a TensorBoard `runs/` folder
      (does not call Detectron2's `EventStorage.put_scalar`) to avoid adding
      metrics that Detectron2 prints in its periodic console output.
    - Pass `to_event_storage=True` to explicitly write into Detectron2's
      `EventStorage` (will be skipped on non-main processes).
    """

    _writer = None
    _writer_lock = threading.Lock()

    @staticmethod
    def _is_dist_avail_and_initialized():
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _is_main_process():
        return (not DetectronLogger._is_dist_avail_and_initialized()) or dist.get_rank() == 0

    @classmethod
    def _get_writer(cls):
        if SummaryWriter is None:
            return None
        if cls._writer is None:
            with cls._writer_lock:
                if cls._writer is None:
                    logdir = os.path.join(os.getcwd(), "runs")
                    os.makedirs(logdir, exist_ok=True)
                    try:
                        cls._writer = SummaryWriter(log_dir=logdir)
                    except Exception:
                        cls._writer = None
        return cls._writer

    @staticmethod
    def log_scalar(name: str, value, step: int = None, to_event_storage: bool = False):
        # Optionally write into Detectron2 EventStorage (may affect console logging)
        if to_event_storage:
            try:
                if not DetectronLogger._is_main_process():
                    return
                storage = get_event_storage()
                if storage is None:
                    return
                if step is None:
                    storage.put_scalar(name, float(value))
                else:
                    storage.put_scalar(name, float(value), smoothing_hint=False)
            except Exception:
                return

        # Default: write to a separate TensorBoard SummaryWriter to avoid
        # polluting Detectron2 EventStorage and triggering its periodic console logs.
        writer = DetectronLogger._get_writer()
        if writer is None:
            return
        try:
            if step is None:
                writer.add_scalar(name, float(value))
            else:
                writer.add_scalar(name, float(value), global_step=int(step))
            writer.flush()
        except Exception:
            return

    @staticmethod
    def print_vars(title: str = "", **kwargs):
        """Print a formatted line of variables to stdout on the main process only.

        Example:
            DetectronLogger.print_vars("[DELTA_CORR]", step=10, gt_in_mean=0.1)
        """
        try:
            if not DetectronLogger._is_main_process():
                return
            parts = []
            if title:
                parts.append(str(title))
            for key, value in kwargs.items():
                parts.append(f"{key}={value}")
            print(" ".join(parts), file=sys.stdout, flush=True)
        except Exception:
            return

