"""
Setup configuration for OMERO Annotation Web Plugin
"""

from setuptools import setup, find_packages

setup(
    name="omero-annotate-web",
    version="0.1.0",
    description="OMERO.web plugin for AI annotation workflow configuration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Maarten Paul",
    author_email="m.w.paul@lumc.nl",
    url="https://github.com/Leiden-Cell-Observatory/omero_annotate_ai",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'omero_annotate_web': [
            'static/omero_annotate_web/css/*',
            'static/omero_annotate_web/js/*',
            'static/omero_annotate_web/js/components/*',
            'templates/omero_annotate_web/*.html',
            'templates/omero_annotate_web/partials/*.html',
        ],
    },
    install_requires=[
        "omero-annotate-ai>=0.1.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Django",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
