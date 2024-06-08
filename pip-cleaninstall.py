import subprocess
import sys

def list_installed_packages():
    """List all installed packages and write them to a file."""
    with open('installed_packages.txt', 'w') as f:
        subprocess.call([sys.executable, '-m', 'pip', 'freeze'], stdout=f)

def uninstall_packages():
    """Uninstall all packages listed in the installed_packages.txt file."""
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-r', 'installed_packages.txt', '-y'])

def clear_pip_cache():
    """Clear the pip cache."""
    subprocess.call([sys.executable, '-m', 'pip', 'cache', 'purge'])

def verify_uninstallation():
    """Verify that all packages have been uninstalled."""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
    print("Installed packages after uninstallation:\n", result.stdout)

if __name__ == "__main__":
    print("Listing installed packages...")
    list_installed_packages()
    
    print("Uninstalling all packages...")
    uninstall_packages()
    
    print("Clearing pip cache...")
    clear_pip_cache()
    
    print("Verifying uninstallation...")
    verify_uninstallation()
