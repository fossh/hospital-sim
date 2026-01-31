"""Project pipelines."""

from kedro.pipeline import Pipeline

from hospital_sim.pipelines.patient_flow import pipeline as patient_flow_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    patient_flow = patient_flow_pipeline.create_pipeline()

    return {
        "__default__": patient_flow,
        "patient_flow": patient_flow,
    }
