"""Hospital Simulation entry point."""

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main():
    configure_project("hospital_sim")
    run = find_run_command("hospital_sim")
    run()


if __name__ == "__main__":
    main()
