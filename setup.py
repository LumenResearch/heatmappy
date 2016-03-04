from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='heatmappy',
    packages=['heatmappy'],
    version='0.2.6',
    description='Draw image heatmaps in python',
    author='Lumen Research',
    author_email='development@lumen-research.com',
    url='https://github.com/LumenResearch/heatmappy',
    download_url='https://github.com/LumenResearch/heatmappy/tarball/0.1.1',
    keywords=['image', 'heatmap', 'heat map'],
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    include_package_data=True,
)
