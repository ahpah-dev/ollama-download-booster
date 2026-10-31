from setuptools import setup, find_packages

setup(
    name="ollama-download-booster",
    version="1.0.0",
    author="ahpah-dev",
    description="Multi-stream download accelerator for Ollama models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahpah-dev/ollama-download-booster",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "urllib3>=2.0.0",
        "customtkinter>=5.2.0",
        "darkdetect>=0.8.0",
    ],
    entry_points={
        "console_scripts": [
            "ollama-booster=ollama_booster.cli:cli_main",
        ],
    },
)
