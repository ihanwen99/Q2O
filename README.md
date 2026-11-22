# A Demonstration of Q2O: Quantum-augmented Query Optimizer

PVLDB, 2025. Paper: https://doi.org/10.14778/3750601.3750691

Q2O is the first quantum-augmented query optimizer that integrates a hybrid quantum-classical approach to solve the join ordering problem in a real database setup. This repository contains the Q2O web demo and the CQM and NL join ordering models it solves on D-Wave Leap.

## Requirements

- Python 3.10
- PostgreSQL with the IMDB database and the `pg_hint_plan` extension
- A D-Wave Leap account with hybrid solver access

## Install

```bash
conda create -n q2o python=3.10
conda activate q2o
pip install -r requirements.txt
dwave setup
python scripts/load_postgres/prepend_headers.py --csv_dir <csv_dir>
bash scripts/load_postgres/load_job_postgres.sh <csv_dir> imdbload
```

Set the database connection in `config.py`.

## Run

```bash
bash scripts/run.sh
python -m backend.run_cqm benchmark/JOB --out cqm_job.csv
```

The first command starts the web demo at http://127.0.0.1:8000/, which runs a query through Q2O and through the default PostgreSQL optimizer side by side. The second command solves the join order of every JOB query with the CQM and writes each join order and its cost to a CSV file.

## Repository layout

```text
Q2O/
├── backend/
│   ├── cqm.py       CQM model
│   ├── nl.py        NL model
│   ├── utils.py     Hints and join order cost
│   ├── postgres.py  PostgreSQL access
│   └── run_cqm.py   CQM over a workload
├── app.py           Web server
├── config.py        Database connection
├── frontend/        Web page
├── benchmark/
│   ├── JOB/         JOB queries
│   └── Synthetic/   Synthetic problems
└── scripts/         Launcher and database loading
```

## Citation

```bibtex
@article{liu2025demonstration,
  author  = {Hanwen Liu and Federico Spedalieri and Ibrahim Sabek},
  title   = {A Demonstration of {Q$^2$O}: Quantum-Augmented Query Optimizer},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {18},
  number  = {12},
  pages   = {5439--5443},
  year    = {2025},
  doi     = {10.14778/3750601.3750691}
}

@inproceedings{liu2025optimizing,
  author    = {Hanwen Liu and Pranshi Saxena and Federico Spedalieri and Ibrahim Sabek},
  title     = {Optimizing Join Orders via Constrained Quadratic Models},
  booktitle = {2025 {IEEE} 41st International Conference on Data Engineering Workshops ({ICDEW})},
  pages     = {295},
  year      = {2025},
  doi       = {10.1109/ICDEW67478.2025.00042}
}
```

## License

GPL-3.0-or-later. See `LICENSE`.
