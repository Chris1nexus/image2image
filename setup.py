from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(name="im2im", install_requires=requirements, packages=find_packages(), include_package_data=True,
      package_data={'im2im': ['models/card/*.md']},)
