import os
import subprocess
import requests
import tempfile
import shutil
from tkinter import messagebox
from tqdm.auto import tqdm


def download_installer(url, dir):
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    file_size = int(response.headers.get('Content-Length', 0))

    with tqdm.wrapattr(response.raw, "read", total=file_size, desc=url) as read:
        dest = os.path.join(dir, "NIMSetup.exe")
        with open(dest, "wb") as f:
            shutil.copyfileobj(read, f)

        return dest


def run_installer(installer_path):
    # Run the installer via powershell with elevated privileges and wait for it to finish
    print("Launching installer...")
    powershell_command = [
        "powershell.exe",
        "-Command",
        f"""
        try {{
            $process = Start-Process '{installer_path}' -Verb RunAs -Wait -PassThru
            exit $process.ExitCode
        }} catch {{
            exit 403  # UAC cancellation
        }}
        """
    ]

    print("Waiting for completion...")
    subprocess.run(powershell_command, check=True)


if os.name == 'nt':
    if messagebox.askquestion("NIM Installer",
                              "Do you want to download and run the NVIDIA NIM installer?\n" +
                              "If you choose not to please ensure you manually setup NVIDIA NIM before attempting to use the node.") == "yes":
        url = "http://127.0.0.1:8080/NIMSetup.exe" # TODO: Find out this URL

        # Download to temp directory so files get deleted
        with tempfile.TemporaryDirectory() as tmpdir:
            installer_path = download_installer(url, tmpdir)
            run_installer(installer_path)

else:
    print("NIM node setup is only supported for Windows")
