from setuptools import find_packages, setup

setup(
    name="ai-metrics",
    version="0.0.1",
    description="A library for basic NLP metric score implementations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hari-hud/ai-metrics",
    author="Hari Hud",
    author_email="hudharibhau@nvidia.com",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "bert_score",
        "sacrebleu",
        "rouge_score",
        "numpy",
    ],
)
