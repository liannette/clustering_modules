import subprocess
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parent.parent

    modules_file = project_dir / "data" / "PRESTO-STAT_modules.txt"
    #n_families = [2000, 4000, 6000, 8000, 10000, 12000]
    n_families = [14000, 16000, 18000, 20000]
    cores = 16
    method = 1  # k-means

    for n in n_families:
        output_dir = project_dir / "results" / f"k_{n}"
        cluster_modules(project_dir, output_dir, modules_file, n, cores, method)


def cluster_modules(project_dir, output_dir, modules_file, n, cores, method):
    script_file = (
        project_dir / "iPRESTO" / "ipresto" / "presto_stat" / "cluster_stat_modules.py"
    )
    log_file = output_dir / "log_families.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "conda",
        "run",
        "-n",
        "ipresto",
        "python",
        str(script_file),
        "--infile",
        str(modules_file),
        "--cores",
        str(cores),
        "--k_clusters",
        str(n),
        "--method",
        str(method),
    ]
    run_command_create_logfile(command, str(log_file))


def run_command_create_logfile(command, log_filepath):
    """Run a command and save the output to a log file.

    Args:
        command (list): The command to be executed.
        log_filepath (str): The path to the log file.

    Returns:
        int: The return code of the command.
    """
    with open(log_filepath, "w") as f:
        process = subprocess.run(
            command, check=True, text=True, stdout=f, stderr=subprocess.STDOUT
        )
    return process.returncode


if __name__ == "__main__":
    main()
