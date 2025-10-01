from pathlib import Path
from setuptools import setup

here = Path(__file__).parent

# Read requirements (ignore blanks/comments)
req_path = here / 'requirements.txt'
if req_path.exists():
    required = [
        line.strip()
        for line in req_path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
else:
    required = []

# Long description
readme_path = here / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

setup(
    name='heatmappy',
    packages=['heatmappy'],
    version='0.3.0',
    description='Draw image heatmaps in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lumen Research',
    author_email='development@lumen-research.com',
    url='https://github.com/LumenResearch/heatmappy',
    keywords=['image', 'heatmap', 'heat map'],
    install_requires=required,
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics',
    ],
    include_package_data=True,
    package_data={'heatmappy': ['assets/*']},
)
