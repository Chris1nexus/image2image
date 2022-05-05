from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setup(name="im2im", install_requires=requirements, 
        dependency_links=['https://download.pytorch.org/whl/lts/1.8/torch_lts.html'],
    packages=find_packages(), 
    include_package_data=True,
    package_data={'im2im': ['models/card/*.md']},)

