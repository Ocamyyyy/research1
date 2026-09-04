import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import experiment1 as base
from task_definition import TaskConfig


base.RESULT_ROOT = Path("results") / "experiments2" / "try1"
base.CASE_ROOT = base.RESULT_ROOT / "cases"


def build_task(N, V, tot, iterr):
    seed = base.stable_seed(N, V, tot, iterr)
    random.seed(seed)
    base.np.random.seed(seed)

    task = TaskConfig()
    task.RP_number = 4
    task.processor_number = task.CPU_number + task.NPU_number + task.RP_number
    task.init_class_vars(base.build_random_dict(N, V, tot), mode="random")
    return task, seed


base.build_task = build_task


def main():
    base.main()


if __name__ == "__main__":
    main()
