"""
Patient Flow Simulation Nodes.

This module implements a discrete-event simulation of hospital patient flow
using SimPy. It models:
- Emergency Department (ED) arrivals and triage
- Inpatient bed allocation
- ICU capacity management
- Operating room scheduling
- Discharge processes
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import simpy

logger = logging.getLogger(__name__)


@dataclass
class HospitalConfig:
    """Hospital configuration parameters."""

    # Capacity
    ed_beds: int = 30
    inpatient_beds: int = 200
    icu_beds: int = 20
    operating_rooms: int = 8

    # Arrival rates (patients per hour)
    ed_arrival_rate: float = 8.0
    scheduled_admission_rate: float = 2.0

    # Service times (hours) - mean values
    ed_triage_time: float = 0.25
    ed_treatment_time: float = 2.5
    inpatient_los: float = 72.0  # Length of stay
    icu_los: float = 48.0
    surgery_duration: float = 2.0

    # Probabilities
    ed_admission_prob: float = 0.25
    icu_admission_prob: float = 0.08
    surgery_prob: float = 0.15

    # Simulation parameters
    simulation_duration: int = 168  # 1 week in hours
    random_seed: int = 42


@dataclass
class PatientMetrics:
    """Metrics collected for each patient."""

    patient_id: int
    arrival_time: float
    patient_type: str  # 'ed' or 'scheduled'
    acuity: int  # 1-5 (1 = most critical)
    ed_wait_time: float = 0.0
    ed_treatment_time: float = 0.0
    admitted: bool = False
    bed_wait_time: float = 0.0
    inpatient_los: float = 0.0
    icu_stay: bool = False
    surgery: bool = False
    discharge_time: float = 0.0
    outcome: str = "pending"


@dataclass
class HospitalMetrics:
    """Aggregated hospital metrics."""

    patients: list = field(default_factory=list)
    ed_queue_lengths: list = field(default_factory=list)
    bed_utilization: list = field(default_factory=list)
    icu_utilization: list = field(default_factory=list)
    or_utilization: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)


class Hospital:
    """Hospital simulation model using SimPy."""

    def __init__(self, env: simpy.Environment, config: HospitalConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Resources (using PriorityResource for priority-based queuing)
        self.ed_beds = simpy.PriorityResource(env, capacity=config.ed_beds)
        self.inpatient_beds = simpy.PriorityResource(env, capacity=config.inpatient_beds)
        self.icu_beds = simpy.PriorityResource(env, capacity=config.icu_beds)
        self.operating_rooms = simpy.PriorityResource(env, capacity=config.operating_rooms)

        # Metrics
        self.metrics = HospitalMetrics()
        self.patient_counter = 0

    def generate_ed_patients(self):
        """Generate ED patient arrivals using Poisson process."""
        while True:
            # Inter-arrival time from exponential distribution
            inter_arrival = self.rng.exponential(1 / self.config.ed_arrival_rate)
            yield self.env.timeout(inter_arrival)

            self.patient_counter += 1
            patient = PatientMetrics(
                patient_id=self.patient_counter,
                arrival_time=self.env.now,
                patient_type="ed",
                acuity=self.rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.30, 0.15]),
            )
            self.env.process(self.ed_patient_flow(patient))

    def generate_scheduled_patients(self):
        """Generate scheduled admission arrivals."""
        while True:
            # Scheduled patients arrive during day hours (8am-6pm)
            hour_of_day = self.env.now % 24
            if 8 <= hour_of_day < 18:
                inter_arrival = self.rng.exponential(1 / self.config.scheduled_admission_rate)
            else:
                # No scheduled admissions at night
                inter_arrival = max(8 - hour_of_day % 24, 0) + self.rng.exponential(0.5)

            yield self.env.timeout(inter_arrival)

            self.patient_counter += 1
            patient = PatientMetrics(
                patient_id=self.patient_counter,
                arrival_time=self.env.now,
                patient_type="scheduled",
                acuity=self.rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3]),
            )
            self.env.process(self.scheduled_patient_flow(patient))

    def ed_patient_flow(self, patient: PatientMetrics):
        """Process ED patient through the system."""
        arrival = self.env.now

        # Wait for ED bed (triage)
        with self.ed_beds.request(priority=patient.acuity) as req:
            yield req
            patient.ed_wait_time = self.env.now - arrival

            # Triage and treatment
            triage_time = self.rng.exponential(self.config.ed_triage_time)
            yield self.env.timeout(triage_time)

            # ED treatment (varies by acuity)
            treatment_multiplier = {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6}
            treatment_time = self.rng.exponential(
                self.config.ed_treatment_time * treatment_multiplier[patient.acuity]
            )
            yield self.env.timeout(treatment_time)
            patient.ed_treatment_time = treatment_time

        # Determine if admission is needed
        admission_prob = self.config.ed_admission_prob * (6 - patient.acuity) / 3
        if self.rng.random() < admission_prob:
            patient.admitted = True
            yield from self.inpatient_flow(patient)
        else:
            patient.outcome = "discharged_ed"
            patient.discharge_time = self.env.now

        self.metrics.patients.append(patient)

    def scheduled_patient_flow(self, patient: PatientMetrics):
        """Process scheduled admission patient."""
        patient.admitted = True
        yield from self.inpatient_flow(patient)
        self.metrics.patients.append(patient)

    def inpatient_flow(self, patient: PatientMetrics):
        """Handle inpatient stay including potential ICU and surgery."""
        bed_request_time = self.env.now

        # Check if ICU needed
        icu_prob = self.config.icu_admission_prob * (6 - patient.acuity) / 3
        needs_icu = self.rng.random() < icu_prob

        if needs_icu:
            # ICU admission
            with self.icu_beds.request(priority=patient.acuity) as req:
                yield req
                patient.bed_wait_time = self.env.now - bed_request_time
                patient.icu_stay = True

                # ICU stay
                icu_los = self.rng.exponential(self.config.icu_los)
                yield self.env.timeout(icu_los)

            # Transfer to regular bed after ICU
            bed_request_time = self.env.now

        # Regular inpatient bed
        with self.inpatient_beds.request(priority=patient.acuity) as req:
            yield req
            if not needs_icu:
                patient.bed_wait_time = self.env.now - bed_request_time

            # Check if surgery needed
            if self.rng.random() < self.config.surgery_prob:
                yield from self.surgery_flow(patient)

            # Inpatient stay
            los = self.rng.exponential(self.config.inpatient_los)
            yield self.env.timeout(los)
            patient.inpatient_los = los

        patient.outcome = "discharged_inpatient"
        patient.discharge_time = self.env.now

    def surgery_flow(self, patient: PatientMetrics):
        """Handle surgical procedure."""
        with self.operating_rooms.request(priority=patient.acuity) as req:
            yield req
            patient.surgery = True

            # Surgery duration
            duration = self.rng.exponential(self.config.surgery_duration)
            yield self.env.timeout(duration)

    def collect_metrics(self):
        """Periodically collect resource utilization metrics."""
        while True:
            self.metrics.timestamps.append(self.env.now)
            self.metrics.ed_queue_lengths.append(len(self.ed_beds.queue))
            self.metrics.bed_utilization.append(
                self.inpatient_beds.count / self.config.inpatient_beds
            )
            self.metrics.icu_utilization.append(self.icu_beds.count / self.config.icu_beds)
            self.metrics.or_utilization.append(
                self.operating_rooms.count / self.config.operating_rooms
            )
            yield self.env.timeout(1)  # Collect every hour


def load_hospital_config(parameters: dict[str, Any]) -> HospitalConfig:
    """Load hospital configuration from parameters."""
    config = HospitalConfig(
        ed_beds=parameters.get("ed_beds", 30),
        inpatient_beds=parameters.get("inpatient_beds", 200),
        icu_beds=parameters.get("icu_beds", 20),
        operating_rooms=parameters.get("operating_rooms", 8),
        ed_arrival_rate=parameters.get("ed_arrival_rate", 8.0),
        scheduled_admission_rate=parameters.get("scheduled_admission_rate", 2.0),
        ed_triage_time=parameters.get("ed_triage_time", 0.25),
        ed_treatment_time=parameters.get("ed_treatment_time", 2.5),
        inpatient_los=parameters.get("inpatient_los", 72.0),
        icu_los=parameters.get("icu_los", 48.0),
        surgery_duration=parameters.get("surgery_duration", 2.0),
        ed_admission_prob=parameters.get("ed_admission_prob", 0.25),
        icu_admission_prob=parameters.get("icu_admission_prob", 0.08),
        surgery_prob=parameters.get("surgery_prob", 0.15),
        simulation_duration=parameters.get("simulation_duration", 168),
        random_seed=parameters.get("random_seed", 42),
    )
    logger.info(f"Loaded hospital config: {config.inpatient_beds} beds, {config.icu_beds} ICU")
    return config


def run_simulation(
    config: HospitalConfig,
    run_id: str,
    sim_number: int,
    num_sims: int,
) -> dict[str, Any]:
    """Run a single hospital simulation."""
    start_time = time.time()

    # Update random seed for this simulation run
    sim_config = HospitalConfig(
        **{**config.__dict__, "random_seed": config.random_seed + sim_number}
    )

    # Create simulation environment
    env = simpy.Environment()
    hospital = Hospital(env, sim_config)

    # Start processes
    env.process(hospital.generate_ed_patients())
    env.process(hospital.generate_scheduled_patients())
    env.process(hospital.collect_metrics())

    # Run simulation
    env.run(until=sim_config.simulation_duration)

    elapsed = time.time() - start_time
    logger.info(
        f"[RUN_ID={run_id}] Simulation {sim_number + 1}/{num_sims} completed in {elapsed:.2f}s"
    )

    return {
        "sim_number": sim_number,
        "patients": hospital.metrics.patients,
        "utilization": {
            "timestamps": hospital.metrics.timestamps,
            "bed_utilization": hospital.metrics.bed_utilization,
            "icu_utilization": hospital.metrics.icu_utilization,
            "or_utilization": hospital.metrics.or_utilization,
            "ed_queue_lengths": hospital.metrics.ed_queue_lengths,
        },
        "elapsed_time": elapsed,
    }


def run_monte_carlo_simulations(
    config: HospitalConfig,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run multiple simulation replications for Monte Carlo analysis."""
    run_id = parameters.get("run_id", os.environ.get("RUN_ID", "local"))
    num_sims = parameters.get("num_sims", 10)
    sim_delay = parameters.get("sim_delay", 0.0)

    logger.info(f"[RUN_ID={run_id}] Starting {num_sims} simulations")

    results = []
    for i in range(num_sims):
        result = run_simulation(config, run_id, i, num_sims)
        results.append(result)

        # Optional delay between simulations (for testing progress tracking)
        if sim_delay > 0:
            time.sleep(sim_delay)

        # Log progress
        progress = (i + 1) / num_sims * 100
        logger.info(f"[RUN_ID={run_id}] Progress: {progress:.1f}% ({i + 1}/{num_sims})")

    logger.info(f"[RUN_ID={run_id}] All simulations completed")
    return results


def aggregate_patient_metrics(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate patient-level metrics across all simulations."""
    all_patients = []

    for result in results:
        sim_num = result["sim_number"]
        for patient in result["patients"]:
            all_patients.append(
                {
                    "sim_number": sim_num,
                    "patient_id": patient.patient_id,
                    "patient_type": patient.patient_type,
                    "acuity": patient.acuity,
                    "arrival_time": patient.arrival_time,
                    "ed_wait_time": patient.ed_wait_time,
                    "ed_treatment_time": patient.ed_treatment_time,
                    "admitted": patient.admitted,
                    "bed_wait_time": patient.bed_wait_time,
                    "inpatient_los": patient.inpatient_los,
                    "icu_stay": patient.icu_stay,
                    "surgery": patient.surgery,
                    "discharge_time": patient.discharge_time,
                    "outcome": patient.outcome,
                    "total_time": patient.discharge_time - patient.arrival_time,
                }
            )

    df = pd.DataFrame(all_patients)
    logger.info(f"Aggregated {len(df)} patient records across {len(results)} simulations")
    return df


def aggregate_utilization_metrics(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate resource utilization metrics."""
    all_utilization = []

    for result in results:
        sim_num = result["sim_number"]
        util = result["utilization"]

        for i, ts in enumerate(util["timestamps"]):
            all_utilization.append(
                {
                    "sim_number": sim_num,
                    "timestamp": ts,
                    "bed_utilization": util["bed_utilization"][i],
                    "icu_utilization": util["icu_utilization"][i],
                    "or_utilization": util["or_utilization"][i],
                    "ed_queue_length": util["ed_queue_lengths"][i],
                }
            )

    df = pd.DataFrame(all_utilization)
    logger.info(f"Aggregated {len(df)} utilization records")
    return df


def _to_native(value: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def calculate_summary_statistics(
    patient_metrics: pd.DataFrame,
    utilization_metrics: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Calculate summary statistics for the simulation results."""
    run_id = parameters.get("run_id", os.environ.get("RUN_ID", "local"))

    # Patient statistics
    ed_patients = patient_metrics[patient_metrics["patient_type"] == "ed"]
    admitted_patients = patient_metrics[patient_metrics["admitted"]]

    summary = {
        "run_id": run_id,
        "num_simulations": _to_native(patient_metrics["sim_number"].nunique()),
        "total_patients": len(patient_metrics),
        "patients_per_sim": len(patient_metrics) / patient_metrics["sim_number"].nunique(),
        # ED metrics
        "ed_patients": len(ed_patients),
        "ed_wait_time_mean": _to_native(ed_patients["ed_wait_time"].mean()),
        "ed_wait_time_p50": _to_native(ed_patients["ed_wait_time"].quantile(0.5)),
        "ed_wait_time_p95": _to_native(ed_patients["ed_wait_time"].quantile(0.95)),
        "ed_treatment_time_mean": _to_native(ed_patients["ed_treatment_time"].mean()),
        # Admission metrics
        "admission_rate": len(admitted_patients) / len(patient_metrics),
        "bed_wait_time_mean": _to_native(admitted_patients["bed_wait_time"].mean()),
        "bed_wait_time_p95": _to_native(admitted_patients["bed_wait_time"].quantile(0.95)),
        "inpatient_los_mean": _to_native(admitted_patients["inpatient_los"].mean()),
        # ICU and surgery
        "icu_rate": _to_native(admitted_patients["icu_stay"].mean()),
        "surgery_rate": _to_native(admitted_patients["surgery"].mean()),
        # Utilization
        "bed_utilization_mean": _to_native(utilization_metrics["bed_utilization"].mean()),
        "bed_utilization_max": _to_native(utilization_metrics["bed_utilization"].max()),
        "icu_utilization_mean": _to_native(utilization_metrics["icu_utilization"].mean()),
        "icu_utilization_max": _to_native(utilization_metrics["icu_utilization"].max()),
        "or_utilization_mean": _to_native(utilization_metrics["or_utilization"].mean()),
        "ed_queue_mean": _to_native(utilization_metrics["ed_queue_length"].mean()),
        "ed_queue_max": _to_native(utilization_metrics["ed_queue_length"].max()),
    }

    logger.info(f"[RUN_ID={run_id}] Summary: {summary['patients_per_sim']:.0f} patients/sim, "
                f"ED wait P95: {summary['ed_wait_time_p95']:.2f}h, "
                f"Bed util: {summary['bed_utilization_mean']:.1%}")

    return summary


def save_results(
    summary: dict[str, Any],
    patient_metrics: pd.DataFrame,
    utilization_metrics: pd.DataFrame,
) -> dict[str, str]:
    """Save simulation results to files."""
    run_id = summary.get("run_id", "local")

    output = {
        "run_id": run_id,
        "summary": summary,
        "patient_count": len(patient_metrics),
        "utilization_records": len(utilization_metrics),
        "status": "completed",
    }

    logger.info(f"[RUN_ID={run_id}] Results saved successfully")
    return output
