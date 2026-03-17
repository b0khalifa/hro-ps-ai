import subprocess
import sys


def run_step(script_name):
    print(f"\n===== Running {script_name} =====")
    result = subprocess.run([sys.executable, script_name], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"✅ {script_name} completed successfully.")


if __name__ == "__main__":
    run_step("feature_engineering.py")
    run_step("prepare_sequences_v2.py")
    run_step("train_arimax_v2.py")
    run_step("train_lstm_v2.py")
    run_step("build_hybrid.py")

    print("\n✅ Full v2 retraining pipeline completed successfully.")