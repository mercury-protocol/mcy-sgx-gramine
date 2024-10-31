from setuptools import setup, find_packages


def parse_requirements() -> list[str]:
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="mcy_dist_ai",
    version="1.0.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=parse_requirements(),
    entry_points={
        "console_scripts": [
            "mcy-split-data=mcy_dist_ai.script.split_data:split_data"
        ],
    },
    author="Peter Berekvolgyi",
    author_email="peter@mercuryprotocol.io",
    description="Mercury leader, worker and watcher",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mercury-protocol/mcy-sgx-gramine",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
