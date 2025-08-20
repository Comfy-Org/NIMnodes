# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------
# Please keep distribution of this code to NVIDIA and partners only
# --------------------------------
# EXAMPLE CODE TO GET NGC API KEY FROM NTS
# This script demonstrates how to get an NGC API key using the NVIDIA GPU Cloud (NGC) Token Service.
# The script uses either nvidia-smi or pynvml library to get the GPU device information 
# This device info is sent to an NVIDIA service for validation to exchange for a token
#
# WARNING - this is only sample code please do not use in production, just as an reference
#
# NOTE - plan to update your client code once PDI is available in NVML and nvidia-smi
#        for now the example will hash a UUID into a pdi-like 64 bit hex string 

import re
import subprocess

import pynvml as nvml
import requests


def get_device_info_nvml():
    try:
        nvml.nvmlInit()
        deviceCount = nvml.nvmlDeviceGetCount()
        deviceInfo = []
        for i in range(deviceCount):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            name = nvml.nvmlDeviceGetName(handle)
            brand = nvml.nvmlDeviceGetBrand(handle)
            architecture = nvml.nvmlDeviceGetArchitecture(handle)
            pciDeviceId = nvml.nvmlDeviceGetPciInfo(handle).pciDeviceId
            pciDeviceId_64bit = format(pciDeviceId, 'X')
            # pdi = 'Unknown' # nvml.nvmlDeviceGetPDI(handle) # TODO - PDI is not available in NVML API yet
            fakePdi = int(hash(uuid) % 2**64) # hash the uuid to a 16 bit integer
            pdi = fakePdi # TODO - just until PDI is available in NVML API
            pdi_64bit = format(pdi, 'X')
            # Map brand and architecture to human-readable strings
            # brand_str = get_brand_name(brand)
            # architecture_str = get_architecture_name(architecture)

            deviceInfo.append({
                'uuid': uuid,
                'pdi': f"0x{pdi_64bit}",
                'name': name,
                'brand': brand,
                'architecture': architecture,
                'pci_device_id': f"0x{pciDeviceId_64bit}",
            })
        nvml.nvmlShutdown()
        print("NVML device info:")
        print(deviceInfo)
        return deviceInfo
    except nvml.NVMLError as e:
        print(f"NVML Error: {e}")
        nvml.nvmlShutdown()
        return []

def get_device_info_smi():
    try:
        output = subprocess.check_output(['nvidia-smi', '-q'], text=True)
        deviceInfo = []
        uuid_pattern = re.compile(r"GPU UUID\s*:\s*(.+)")
        pdi_pattern = re.compile(r"PDI\s*:\s*(.+)")
        name_pattern = re.compile(r"Product Name\s*:\s*(.+)")
        brand_pattern = re.compile(r"Product Brand\s*:\s*(.+)")
        arch_pattern = re.compile(r"Product Architecture\s*:\s*(.+)")
        pci_pattern = re.compile(r"Device Id\s*:\s*(.+)")

        matches = {
            "uuid": uuid_pattern.search(output),
            "pdi": pdi_pattern.search(output),
            "name": name_pattern.search(output),
            "brand": brand_pattern.search(output),
            "architecture": arch_pattern.search(output),
            "pci_device_id": pci_pattern.search(output),
        }

        # TODO - just until PDI is available in NVML API
        if not matches["pdi"]:
            fakePdi = int(hash(matches["uuid"].group(1).strip()) % 2**64) # hash the uuid to a 16 bit integer for testing
            pdi_64bit = format(fakePdi, 'X')
            matches["pdi"] = re.match(r"(.+)", f"0x{pdi_64bit}")

        deviceInfo.append({k: (m.group(1).strip() if m else "Unknown") for k, m in matches.items()})
        print("SMI device info:")
        print(deviceInfo)
        return deviceInfo
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi error: {e}")
        return []

def validate_device_info(deviceInfo):
    matchInName = ['RTX', 'OTHER DEVICE NAME TEST']
    for device in deviceInfo:
        if any(keyword in device.get('name', '') for keyword in matchInName):
            print(f"Valid device name {device['name']} found")
            return True
    matchInBrandString = ['GeForce','5']
    for device in deviceInfo:
        if any(keyword in str(device.get('brand', '')) for keyword in matchInBrandString):
            print(f"Valid device brand {device['brand']} found")
            return True
    print("No valid device found in the list")
    return False

def get_ngc_key_from_device_info(deviceInfo):
    # Optional - check env for vars
    # if NGC_API_KEY or NGC_CLI_API_KEY decide if the client should use that instead
    # the key will still need to be setup properly with Catalog Access, and the owner will need an NVAIE sub  # noqa: E501
    # or be a member of the NVIDIA developer program (https://developer.nvidia.com/developer-program)
    # apiKey = os.environ.get('NGC_API_KEY')
    # if apiKey:
    #     print("Using the API key from the environment variable instead of calling API")  # noqa: E501
    #     return apiKey

    ngcKeyServiceUrl = 'https://nts.ngc.nvidia.com/v1/token'
    payload = {
        "client_id": "comfyui",
        "pdi": deviceInfo[0].get('pdi', 'Unknown') if deviceInfo else 'Unknown',
        "access_policy_name": "nim-dev",
        "device": deviceInfo[0] if deviceInfo else {}
    }
    print(f"debug Payload: {payload}")
    try:
        response = requests.post(ngcKeyServiceUrl, headers={'Accept': 'application/json'}, json=payload)  # noqa: E501
        print(f"Response: {response}")
        response.raise_for_status()
        keyData = response.json()
        print(f"Key data: {keyData}")
        return keyData.get('access_token')
    except requests.RequestException as e:
        print(f"Error fetching API key: {e}")
        return None

def get_ngc_key():
    # two different examples of getting device info 
    # note: branch/architecture, nvidia-smi gives strings, NVML gives ints
    deviceInfoNvml = get_device_info_nvml()
    deviceInfoSmi = get_device_info_smi()

    # client check option - TBD if needed by client development team
    # if not validate_device_info(deviceInfoNvml):
    #     print("Device info validation failed - exiting")
    #     exit(1)

    key = get_ngc_key_from_device_info(deviceInfoNvml)
    if key:
        print(f"Key: {key}")
        return key
    else:
        raise Exception("Error getting NGC API key. Please follow the instructions in the README to set up your NGC API key.")  # noqa: E501


if __name__ == "__main__":
    # deviceInfoNvml = get_device_info_nvml()
    # deviceInfoSmi = get_device_info_smi()
    # assert deviceInfoNvml[0]["pdi"] == deviceInfoSmi[0]["pdi"]
    # key = get_ngc_key_from_device_info(deviceInfoSmi)
    get_ngc_key()
