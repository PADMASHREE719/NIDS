import os
import sys
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_step(step_num, title):
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.BLUE}STEP {step_num}: {title}{Colors.END}")
    print("="*70)

def run_command(cmd, description):
    print(f"\n{Colors.YELLOW}> {description}{Colors.END}")
    print(f"  Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{Colors.GREEN}Success!{Colors.END}")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"{Colors.RED}Error!{Colors.END}")
        if result.stderr:
            print(result.stderr)
        return False

def check_environment():
    print_step(0, "Checking Environment")
    required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'yaml', 'joblib', 'tqdm', 'matplotlib', 'seaborn']
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                __import__('yaml')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  OK {package}")
        except ImportError:
            print(f"  MISSING {package}")
            missing.append(package)
    if missing:
        print(f"\nInstalling missing packages...")
        pip_names = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}
        for pkg in missing:
            pip_name = pip_names.get(pkg, pkg)
            run_command(f"pip install {pip_name}", f"Installing {pip_name}")
    return True

def setup_directories():
    print_step(1, "Setting Up Directories")
    directories = ['data/artifacts', 'models', 'results', 'data/flows', 'data/pcap_samples']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    return True

def check_data():
    print_step(2, "Checking Data Files")
    csv_dir = Path("data/raw/CICIDS2017")
    if not csv_dir.exists():
        print(f"ERROR: Directory not found: {csv_dir}")
        return False
    csv_files = list(csv_dir.glob("*.csv"))
    pcap_files = list(csv_dir.glob("*.pcap"))
    print(f"\n  Found {len(csv_files)} CSV files:")
    for f in csv_files[:5]:
        print(f"    - {f.name}")
    if len(csv_files) > 5:
        print(f"    ... and {len(csv_files) - 5} more")
    print(f"\n  Found {len(pcap_files)} PCAP files")
    if len(csv_files) == 0 and len(pcap_files) == 0:
        print(f"ERROR: No data files found!")
        return False
    return True

def run_preprocessing():
    print_step(3, "Preprocessing Data")
    if not run_command("python src/merge_align.py", "Computing master feature list"):
        return False
    if not run_command("python src/preprocess.py", "Building and preprocessing dataset"):
        return False
    return True

def run_training():
    print_step(4, "Training Model")
    if not run_command("python src/train.py", "Training CapsNet-BiLSTM model"):
        return False
    return True

def run_evaluation():
    print_step(5, "Evaluating Model")
    if not run_command("python src/evaluate.py", "Evaluating model performance"):
        return False
    return True

def show_results():
    print_step(6, "Project Complete!")
    print("\nYOUR NIDS PROJECT IS COMPLETE!\n")
    print("Generated Artifacts:")
    artifacts = [
        ("data/artifacts/X_all.npy", "Preprocessed features"),
        ("data/artifacts/y_all.npy", "Labels"),
        ("data/artifacts/scaler.pkl", "Feature scaler"),
        ("data/artifacts/label_encoder.pkl", "Label encoder"),
        ("models/best.pt", "Trained model"),
        ("results/confusion_matrix.png", "Confusion matrix")
    ]
    for path, description in artifacts:
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024
            print(f"  OK {path:40s} ({size:.1f} KB) - {description}")
        else:
            print(f"  -- {path:40s} - Not found")
    print(f"\nNext Steps:")
    print("  1. Check confusion matrix: results/confusion_matrix.png")
    print("  2. Test live inference: python src/live_inference.py")
    print("  3. Start API server: python src/server.py")
    print(f"\nYour model is ready to detect attacks!")

def main():
    print("\n" + "="*70)
    print("     NETWORK INTRUSION DETECTION SYSTEM")
    print("           Complete Project Runner")
    print("="*70)
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"\nProject directory: {project_root.absolute()}\n")
    try:
        steps = [
            ("Environment Check", check_environment),
            ("Directory Setup", setup_directories),
            ("Data Verification", check_data),
            ("Preprocessing", run_preprocessing),
            ("Model Training", run_training),
            ("Model Evaluation", run_evaluation),
        ]
        for step_name, step_func in steps:
            if not step_func():
                print(f"\nFailed at: {step_name}")
                print(f"\nPlease fix the error above and run again.")
                return False
        show_results()
        return True
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)