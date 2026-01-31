"""Patient flow simulation pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_patient_metrics,
    aggregate_utilization_metrics,
    calculate_summary_statistics,
    load_hospital_config,
    run_monte_carlo_simulations,
    save_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the patient flow simulation pipeline."""
    return pipeline(
        [
            node(
                func=load_hospital_config,
                inputs="parameters",
                outputs="hospital_config",
                name="load_config",
            ),
            node(
                func=run_monte_carlo_simulations,
                inputs=["hospital_config", "parameters"],
                outputs="simulation_results",
                name="run_simulations",
            ),
            node(
                func=aggregate_patient_metrics,
                inputs="simulation_results",
                outputs="patient_metrics",
                name="aggregate_patients",
            ),
            node(
                func=aggregate_utilization_metrics,
                inputs="simulation_results",
                outputs="utilization_metrics",
                name="aggregate_utilization",
            ),
            node(
                func=calculate_summary_statistics,
                inputs=["patient_metrics", "utilization_metrics", "parameters"],
                outputs="summary_statistics",
                name="calculate_summary",
            ),
            node(
                func=save_results,
                inputs=["summary_statistics", "patient_metrics", "utilization_metrics"],
                outputs="final_output",
                name="save_results",
            ),
        ]
    )
