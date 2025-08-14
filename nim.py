import os
import socket
import subprocess
from enum import Enum
from pathlib import Path
from .ngc import get_ngc_key
import time
import re
import atexit
from typing import List
import sys
import json
import requests
import threading
import queue

TIME_OUT = 1800

class ModelType(Enum):
    FLUX_DEV = "FLUX_DEV"
    FLUX_CANNY = "FLUX_CANNY"
    FLUX_DEPTH = "FLUX_DEPTH"
    FLUX_SCHNELL = "FLUX_SCHNELL"
    FLUX_KONTEXT = "FLUX_KONTEXT"
    SD35L_BASE = "SD35L_BASE"
    SD35L_CANNY = "SD35L_CANNY"
    SD35L_DEPTH = "SD35L_DEPTH"


class OffloadingPolicy(Enum):
    NONE = "None"
    SYS = "System RAM"
    DISK = "Disk"
    DEFAULT = "Default"

class NIMManager:
    '''
    This class is responsible for managing the NIM containers.

    Supports: 
    - SDXL
    - FLUX
    '''

    # Registry paths for different model types
    MODEL_REGISTRY: dict[ModelType, str] = {
        ModelType.FLUX_DEV: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_CANNY: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_DEPTH: "nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0",
        ModelType.FLUX_SCHNELL: "nvcr.io/nim/black-forest-labs/flux.1-schnell:1.0.0",
        ModelType.FLUX_KONTEXT: "nvcr.io/nim/black-forest-labs/flux.1-kontext-dev:1.0.0",
        ModelType.SD35L_BASE: "nvcr.io/nim/stabilityai/stable-diffusion-3.5-large:1.0.0",
        ModelType.SD35L_CANNY: "nvcr.io/nim/stabilityai/stable-diffusion-3.5-large:1.0.0",
        ModelType.SD35L_DEPTH: "nvcr.io/nim/stabilityai/stable-diffusion-3.5-large:1.0.0",
    }
    PORT = 5000

    def __init__(self):
        self._nim_server_proc_dict: dict[ModelType, dict] = {}
        self.api_key = get_ngc_key()
        self.cache_path = self._get_cache_path()
        atexit.register(self.cleanup)
        self.cmd_prefix = ""
        if os.name == 'nt':
            if self.is_wsl_distribution_installed(distro_name="NVIDIA-Workbench"):
                self.cmd_prefix = "wsl -d NVIDIA-Workbench -- "


    def get_wsl_distributions(self):
        try:
            result = subprocess.run("wsl --list", capture_output=True, shell=True)

            if result.returncode != 0:
                print("Failed to retrieve WSL distributions.")
                return []
            
            distributions = [d.decode('utf-8').replace("\x00", "").strip() for d in result.stdout.splitlines()[1:] if len(d) > 1]
            return distributions

        except Exception as e:
            print(f"Error: {e}")
            return []


    def is_wsl_distribution_installed(self, distro_name):
        wsl_distributions = self.get_wsl_distributions()
        for d in wsl_distributions:
            if distro_name in d or distro_name == d:
                return True
        return False

    def check_folder_in_wsl(self, folder_path):
        """
        Checks if a folder exists in a specified WSL distribution.

        Args:
            distro_name (str): The name of the WSL distribution.
            folder_path (str): The absolute path to the folder within the distro.

        Returns:
            bool: True if the folder exists, False otherwise.
        """
        try:
            # The 'test -d' command returns a 0 exit code for success (folder exists)
            # and a non-zero exit code for failure (folder doesn't exist).
            result = subprocess.run(
                ["wsl.exe", "-d", "NVIDIA-Workbench", "test", "-d", folder_path],
                check=True,  # This will raise an exception if the command fails
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True
        except subprocess.CalledProcessError:
            # A non-zero return code means the folder doesn't exist.
            return False
        except FileNotFoundError:
            print("Error: wsl.exe not found. Please ensure it's in your system PATH.")
            return False

    def _get_cache_path(self, wsl_path=True) -> str:
        home = Path.home()
        if os.name == "nt" and wsl_path:
            home = Path("/mnt/" + home.parts[0][0].lower() + "/" + "/".join(home.parts[1:]))
        cache_path = home / "nimcache/{model_name}/latest/.cache"
        
        if wsl_path:
            return cache_path.as_posix()
        return str(cache_path)


    def _run_cmd(self, cmd: str, err_msg: str = "Unknown") -> List[str]:
        cmd = self.cmd_prefix + cmd
        print(f'The command from _run_cmd: {cmd}')
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True)

        if result.returncode != 0:
            error_msg = (
                f"Failed to {err_msg}:\n"
                f"stdout: {result.stdout.decode('utf-8')}\n"
                f"stderr: {result.stderr.decode('utf-8')}"
            )
            raise Exception(error_msg)
        return result.stdout.decode("utf-8").split("\n")
    

    def _run_proc(self, cmd: str):
        cmd = self.cmd_prefix + cmd
        print(f'The command from run_Proc: {cmd}')
        run_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        if run_process:
            return run_process
        else:
            raise Exception("Launching process failed")

        
    def _setup_directories(self, model_name: ModelType) -> None:
        """Create necessary directories for NIM cache"""
        cache_path = self.cache_path.format(model_name=model_name.value)

        native_path = self._get_cache_path(wsl_path=False).format(model_name=model_name.value)

        if not self.check_folder_in_wsl('~/.cache/nim'):
            self._run_cmd('mkdir -p ~/.cache/nim')
            self._run_cmd('chmod 777 -R ~/.cache/nim')

        if os.path.exists(native_path):
            return
        os.makedirs(native_path, exist_ok=True)    
        # Set permissions using WSL
        chmod_command = f"chmod 777 -R {cache_path}"
        self._run_cmd(chmod_command, "setup cache directory")
        print(f"Directory setup completed for {model_name.value}")


    def pull_nim_image(self, model_name: ModelType) -> None:
        """Pull NIM image from the registry"""
        command = f"podman login --username '$oauthtoken' --password {self.api_key} nvcr.io"
        self._run_cmd(command)

        registry_path = self.MODEL_REGISTRY[model_name]
        command = f"podman pull {registry_path}"
        # self._run_cmd(command, "pull NIM")
        process = self._run_proc(command)
        while True:
            output = process.stderr.readline()
            exit_code = process.poll()
            if exit_code != None:
                if exit_code == 0:
                    break
                else:
                    raise Exception("Failed to pull the image")
            if output: # podman pull image logs
                sys.stdout.write(f"\r{output.decode("utf-8").strip()}\n")
                sys.stdout.flush()
        print("Image has been pulled")

    def get_running_container_info(self):
        cmd = self.cmd_prefix + f"podman container ls -a --format json"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print("Error fetching Podman containers")
            return []
        containers_json = json.loads(result.stdout.decode("utf-8"))
        containers_data = {}
        for container in containers_json:
            if "Names" in container and "Ports" in container:
                id = container["Id"]
                image = container["Image"]
                for i in range(len(container["Names"])):
                    name = container["Names"][i]
                    ports = []
                    for port_info in container["Ports"]:
                        ports.append(port_info.get("host_port"))
                    containers_data[name] = {"ports": ports, "id": id, "image": image}
        return containers_data
    

    def is_nim_running(self, model_name: ModelType):
        containers_data = self.get_running_container_info()
        if model_name.value in containers_data:
            if model_name in self._nim_server_proc_dict.keys():
                return True
            self.stop_nim(model_name, force=True)
        return False

    def _get_variant(self, model_name: ModelType):
        if model_name.value.endswith("CANNY"):
            return "canny"
        elif model_name.value.endswith("DEPTH"):
            return "depth"
        elif model_name.value.endswith("SCHNELL"):
            return "base"
        elif model_name.value.endswith("KONTEXT"):
            return "base"
        elif model_name.value.endswith("BASE"):
            return "base"
        else:
            return "base"

    def start_nim_container(self, model_name: ModelType, offloading_policy: OffloadingPolicy, hf_token: str = "") -> None:
        """Start a NIM container with the specified configuration"""
        if self.is_nim_running(model_name):
            print(f"NIM for {model_name.value} is already running...")
            return

        self._setup_directories(model_name)
        #cache_path = self.cache_path.format(model_name=model_name.value)
        cache_path = '~/.cache/nim'

        # Check if port is already in use
        port = self.PORT + len(self._nim_server_proc_dict)

        while self.is_port_in_use(port):
            port += 1
        
        variant = self._get_variant(model_name)
        
        # show start container logs
        command = (
            f"podman run --rm "
            f"--device=nvidia.com/gpu=all "
            f"--name={model_name.value} "
            f"--shm-size=16GB "
            f"-e NGC_API_KEY={self.api_key} "
            f"-v {cache_path}:/opt/nim/.cache "
            f"-e NIM_RELAX_MEM_CONSTRAINTS=1 "
            f"-e NIM_OFFLOADING_POLICY={offloading_policy.replace(" ", "_").lower()} "
            f"-e NIM_MODEL_VARIANT={variant} "
            f"-e HF_TOKEN={hf_token} "
            f"-p {port}:8000 "
            f"{self.MODEL_REGISTRY[model_name]}"
        )
        print(command)
        process = self._run_proc(command)
        self._nim_server_proc_dict[model_name] = {"port": port, "id": None}

        invoke_url = f"http://localhost:{port}/v1/health/ready"
        start_time = time.time()

        # read podman container logs
        def read_process_stdout(process, output_queue):
            """ read stdout and put on the queue """
            for line in iter(process.stdout.readline, b''):
                output_queue.put(line.decode('utf-8'))
            process.stdout.close()

        def read_process_stderr(process, output_queue):
            """ read stderr and put on the queue """
            for line in iter(process.stderr.readline, b''):
                output_queue.put(line.decode('utf-8'))
            process.stderr.close()
        
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        stdout_thread = threading.Thread(target=read_process_stdout, args=(process, stdout_queue), daemon=True)
        stderr_thread = threading.Thread(target=read_process_stderr, args=(process, stderr_queue), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        while True:
            time.sleep(1)
            try:
                log_line = stdout_queue.get_nowait()
                if log_line:
                    sys.stdout.write(log_line)
                    sys.stdout.flush()
            except queue.Empty:
                pass  
            try:
                log_line = stderr_queue.get_nowait()
                if log_line:
                    sys.stdout.write(log_line)
                    sys.stdout.flush()
            except queue.Empty:
                pass  
            try:
                response = requests.get(invoke_url)
                if response.status_code == 200:
                    wait_time = time.time() - start_time
                    print(f"NIM service endpoint is up and running after waiting {round(wait_time)} seconds!")
                    return
            except:
                pass

            if time.time() - start_time > TIME_OUT:
                raise TimeoutError("NIM Server did not start within the specified timeout.")


    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
        

    def get_port(self, model_name: ModelType) -> int:
        if not self.is_nim_running(model_name):
            raise Exception(f"NIM {model_name.value} is not running. Please ensure that you have started the NIM container via podman.")
        if model_name in self._nim_server_proc_dict:
            return self._nim_server_proc_dict[model_name]["port"]
        containers_data = self.get_running_container_info()
        return containers_data[model_name.value]["port"]


    def deploy_nim(self, model_name: ModelType, offloading_policy: OffloadingPolicy, hf_token: str) -> None:
        """Deploy a NIM model with all necessary setup"""
        # Setup directories
        self._setup_directories(model_name)
            
        # Pull image
        self.pull_nim_image(model_name)
            
        # Start container
        self.start_nim_container(
            model_name,
            offloading_policy,
            hf_token
        )
    

    def stop_nim(self, model_name: ModelType, force: bool = False) -> None:
        if not force:
            if not self.is_nim_running(model_name):
                print(f"NIM {model_name.value} is already stopped.")
                return
        command = f"podman stop {model_name.value}"
        self._run_cmd(command, f"stop NIM {model_name.value}")
        if model_name in self._nim_server_proc_dict:
            self._nim_server_proc_dict.pop(model_name)
        print(f"Stopped NIM {model_name.value}")


    def cleanup(self) -> None:
        nims = list(self._nim_server_proc_dict.keys())
        for model in nims:
            try:
                self.stop_nim(model)
            except RuntimeError as e:
                if "new thread" in str(e):
                    print("Handled RuntimeError: can't create new thread at interpreter shutdown")
            finally:
                command = self.cmd_prefix + f"podman stop {model.value}"
                try:
                    subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True  # Detach from the Python interpreter
                    )
                    print(f"Stopping NIM {model.value}")
                except Exception as e:
                    print(f"Error stopping {model.value}: {e}")
            

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


    def __del__(self):
        self.cleanup()



if __name__ == "__main__":
    model_name = ModelType.FLUX_DEV
    registry_path = NIMManager.MODEL_REGISTRY[model_name]

    manager = NIMManager()
    manager.deploy_nim(model_name)
    manager.stop_nim(model_name)
