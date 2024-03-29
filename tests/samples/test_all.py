import glob
import os.path
import subprocess


def test_all() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(root_dir)
    sample_dirs = glob.glob(os.path.join(root_dir, "samples", "*"))
    for sample_dir in sample_dirs:
        if not os.path.isdir(sample_dir):
            continue
        if os.path.basename(sample_dir) == "__pycache__":
            continue
        cmd = [
            "python",
            f"{sample_dir}/main.py",
            "--config",
            f"{sample_dir}/config.json",
            "--seed",
            "1",
        ]
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] += f";{root_dir}"
        else:
            env["PYTHONPATH"] = root_dir
        run = subprocess.run(
            cmd,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            shell=("OS" in os.environ and os.environ["OS"].startswith("Windows")),
        )
        if run.returncode != 0 or len(run.stdout) == 0:
            raise RuntimeError(
                f"Error: {' '.join(cmd)}\n{str(run.stdout)}\n{str(run.stderr)}"
            )
