# Hospital Patient Flow Simulation

A discrete-event simulation of hospital patient flow for capacity planning and resource optimization. Built with [Kedro](https://kedro.org/) and [SimPy](https://simpy.readthedocs.io/).

## Overview

This simulation models:

- **Emergency Department (ED)**: Patient arrivals, triage, and treatment
- **Inpatient Beds**: Admission from ED and scheduled admissions
- **ICU**: Critical care capacity management
- **Operating Rooms**: Surgical procedure scheduling
- **Discharge**: Patient flow through the system

## Use Cases

- **Capacity Planning**: Determine optimal bed counts for different units
- **Wait Time Analysis**: Identify bottlenecks causing patient delays
- **Resource Optimization**: Balance staffing and equipment allocation
- **Scenario Analysis**: Test impact of demand surges (e.g., flu season)

## Installation

```bash
# Clone the repository
git clone https://github.com/fossh/hospital-sim.git
cd hospital-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Run default simulation (10 replications, 1 week each)
kedro run

# Run with custom parameters
kedro run --params="num_sims=20,simulation_duration=336"

# Run with external run_id (for integration with simctl)
kedro run --params="run_id=$RUN_ID,num_sims=10,sim_delay=0.2"
```

## Configuration

### Parameters (`conf/base/parameters.yml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sims` | 10 | Number of Monte Carlo simulations |
| `sim_delay` | 0.5 | Delay between sims (for progress tracking) |
| `ed_beds` | 30 | Emergency department bed capacity |
| `inpatient_beds` | 200 | General inpatient beds |
| `icu_beds` | 20 | Intensive care unit beds |
| `operating_rooms` | 8 | Operating room count |
| `ed_arrival_rate` | 8.0 | ED patients per hour |
| `simulation_duration` | 168 | Hours to simulate (168 = 1 week) |

### Scenario Examples

**High Demand Scenario:**
```bash
kedro run --params="ed_arrival_rate=12.0,scheduled_admission_rate=4.0"
```

**Capacity Expansion Test:**
```bash
kedro run --params="inpatient_beds=250,icu_beds=30"
```

**Long-term Simulation:**
```bash
kedro run --params="simulation_duration=720,num_sims=50"  # 30 days, 50 runs
```

## Output

Results are saved to:

- `data/07_model_output/patient_metrics.csv` - Individual patient records
- `data/07_model_output/utilization_metrics.csv` - Resource utilization over time
- `data/08_reporting/summary_statistics.json` - Aggregated statistics
- `data/08_reporting/final_output.json` - Run completion status

### Key Metrics

- **ED Wait Time (P95)**: 95th percentile wait for ED bed
- **Bed Wait Time**: Time from admission decision to bed assignment
- **Bed Utilization**: Average and peak occupancy rates
- **ICU Utilization**: Critical care capacity usage
- **Queue Lengths**: Number of patients waiting at each stage

## Integration with simctl

This project is designed to work with the `simctl` CLI:

```bash
# Create simulator
simctl simulator create --repo https://github.com/fossh/hospital-sim \
  --name "Hospital Sim" \
  --run-command "kedro run --params run_id=\$RUN_ID,num_sims=10,sim_delay=0.2"

# Build AMI
simctl ami build <simulator-id> --wait

# Run simulation
simctl run create <simulator-id> --event "capacity-test"
simctl run start <run-id> --watch
```

## Project Structure

```
hospital-sim/
├── conf/
│   ├── base/
│   │   ├── catalog.yml      # Data catalog
│   │   ├── parameters.yml   # Simulation parameters
│   │   └── logging.yml      # Logging configuration
│   └── local/               # Local overrides (not in git)
├── data/
│   └── ...                  # Output directories
├── src/
│   └── hospital_sim/
│       ├── pipelines/
│       │   └── patient_flow/
│       │       ├── nodes.py     # Simulation logic
│       │       └── pipeline.py  # Pipeline definition
│       ├── __init__.py
│       ├── __main__.py
│       ├── pipeline_registry.py
│       └── settings.py
├── pyproject.toml
└── README.md
```

## Simulation Model

### Patient Flow

```
ED Arrival ──► Triage ──► ED Treatment ──┬──► Discharge
                                         │
                                         ▼
Scheduled ──────────────────────► Inpatient Bed ──┬──► Discharge
Admission                              │          │
                                       ▼          │
                                   ICU Bed ───────┘
                                       │
                                       ▼
                                Operating Room
```

### Assumptions

- Patient arrivals follow Poisson process
- Service times are exponentially distributed
- Priority queuing based on acuity (1=critical, 5=minor)
- Resources are shared across all patient types

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
ruff check src/
```

## License

MIT
