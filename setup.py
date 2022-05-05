import os
import sys
import setuptools
import subprocess


def remove_requirements(requirements, name, replace=''):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements

sys_platform = sys.platform




with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

#'''
requirements = remove_requirements(requirements,'torch')
requirements = remove_requirements(requirements,'torchvision')
requirements = remove_requirements(requirements,'torchvision')

TORCH_VERSION='1.8.1'
TORCH_HW_VERSION='cpu'

print('Trying to install pytorch and torchvision!')
code = 1
try:
    code = subprocess.call(['pip', 'install', f'torch==={TORCH_VERSION}+{TORCH_HW_VERSION}', 
                    f'torchvision===0.8.1+{TORCH_HW_VERSION}', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
    if code != 0:
        raise Exception('Torch and torchvsion instalation failed !')
except:
    try:
        code = subprocess.call(['pip3', 'install', f'torch==={TORCH_VERSION}+{TORCH_HW_VERSION}', 
                    f'torchvision===0.8.1+{TORCH_HW_VERSION}', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise Exception('Torch and torchvision installation failed !')
    except:
        print('Failed to install pytorch, please install pytorch and torchvision manually by following the simple instructions over at: https://pytorch.org/get-started/locally/')
if code == 0:
    print('Successfully installed pytorch and torchvision CPU version! (If you need the GPU version, please install it manually, checkout the mindsdb docs and the pytorch docs if you need help)')
#'''
with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setuptools.setup(name="im2im", install_requires=requirements, 
        dependency_links=['https://download.pytorch.org/whl/torch_stable.html',],
    packages=setuptools.find_packages(), 
    include_package_data=True,
    package_data={'im2im': ['models/card/*.md'],})