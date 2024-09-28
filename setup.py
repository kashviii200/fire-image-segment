import subprocess
import sys

def upgrade_pip():
    try:
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip==24.2"])
        print("pip upgraded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade pip: {e}")
        sys.exit(1)

def install_requirements():
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Step 1: Upgrade pip
    upgrade_pip()

    # Step 2: Install dependencies
    install_requirements()
