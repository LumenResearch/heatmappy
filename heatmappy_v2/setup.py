from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='heatmappy2',
    packages=find_packages(),
    version='0.1.0',
    description='Draw image and video heatmaps in python',
    author='Lumen Research',
    author_email='development@lumen-research.com',
    url='https://github.com/LumenResearch/heatmappy',
    keywords=['image', 'video', 'heatmap', 'heat map', 'opencv'],
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    include_package_data=True,
)
