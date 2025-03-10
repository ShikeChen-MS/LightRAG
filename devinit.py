import os
import sys
import subprocess
import platform
import venv


def check_python_version():
    """
    Check Python version to ensure it is 3.11 or higher.
    """
    if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] < 11
    ):
        print(
            "Minimum version of Python 3.11 is required for this project. Please run it with an appropriate version."
        )
        sys.exit(1)


def create_venv(venv_name: str):
    """
    Creating a virtual environment with the given name and Python version.
    :param venv_name: desired name of the virtual environment
    :param python_version: desired Python version (3.11 or higher)
    """
    print(f"Creating virtual environment '{venv_name}'...")
    if platform.system() == "Windows":
        upgrade_command = f"{venv_name}\\Scripts\\python -m pip install --upgrade pip"
    else:
        upgrade_command = f"{venv_name}/bin/python -m pip install --upgrade pip"
    try:
        venv.create(venv_name, with_pip=True)
        print("Upgrading pip to the latest version...")
        subprocess.run(upgrade_command, check=True, shell=True)
        print(f"Virtual environment '{venv_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        sys.exit(1)


def install_requirements(venv_name: str, requirements_file: str):
    """
    Installing requirements from a file into a virtual environment.
    :param venv_name: name of the virtual environment
    :param requirements_file: path to the requirements file
    """
    print(
        f"Installing requirements from '{requirements_file}' into virtual environment '{venv_name}'..."
    )
    if platform.system() == "Windows":
        command = f"{venv_name}\\Scripts\\pip install -r {requirements_file}"
    else:
        command = f"{venv_name}/bin/pip install -r {requirements_file}"
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Requirements from '{requirements_file}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        sys.exit(1)


def check_file_existence(filename: str) -> bool:
    """
    Check if a file exists in the current directory.
    :param filename: name of the file to check
    :return: True if the file exists, False otherwise
    """
    if os.path.isfile(filename):
        print(f"Requirement file '{filename}' detected in the current directory.")
        return True
    print(f"Requirements file '{filename}' not found in the current directory.")
    return False


if __name__ == "__main__":
    check_python_version()

    venv_name = ".venv"
    requirements_file = [
        "requirements.txt",
        "./devrequirements.txt",
    ]

    create_venv(venv_name)
    req_install = True
    for requirements_file in requirements_file:
        if check_file_existence(requirements_file):
            install_requirements(venv_name, requirements_file)
        else:
            print(
                f"Requirements file '{requirements_file}' not found. Continue without it..."
            )
            req_install = False
    if req_install:
        print("All requirements installed successfully.")
        sys.exit(0)
    print(
        "Virtual Environment created successfully, but not all requirements were installed..."
    )
    sys.exit(1)
