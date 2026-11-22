import argparse
import csv
import time
from pathlib import Path

from backend.cqm import build_cqm, load_statistics, solve_cqm
from backend.utils import (
    compute_db_cost,
    list_folders_by_number,
)

CQM_TIME_LIMIT_S = 5


def main():
    parser = argparse.ArgumentParser(
        description="Solve left-deep join orders with the CQM and report their cost."
    )
    parser.add_argument(
        "workload", type=Path, help="folder with one sub-folder per query"
    )
    parser.add_argument("--runs", type=int, default=5, help="solver calls per query")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "run", "join_order", "cost", "seconds", "qpu_ms"])
        for folder in list_folders_by_number(args.workload):
            cards, sels = load_statistics(folder)
            for run in range(args.runs):
                start = time.time()
                cqm = build_cqm(cards, sels)
                order, qpu_ms = solve_cqm(cqm, CQM_TIME_LIMIT_S)
                seconds = time.time() - start
                order = [int(i) for i in order]
                cost = compute_db_cost(order, cards, sels)
                writer.writerow(
                    [
                        Path(folder).name,
                        run,
                        " ".join(map(str, order)),
                        cost,
                        f"{seconds:.3f}",
                        qpu_ms,
                    ]
                )
                f.flush()
                print(Path(folder).name, run, order, cost)


if __name__ == "__main__":
    main()
