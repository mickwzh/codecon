# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='codecon',
    version='0.1',
    description='A machine learning library for economists',
    author='Zhaohui Wang',
    author_email='mickwang@connect.hku.hk',
    packages=find_packages(),
    package_data = {
        'codecon': ['data/cn_stopwords.txt'],  # Specify the data file
    },
    long_description= long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    install_requires=[
        'pip',
        'pandas',
        'numpy<2',
        'scikit-learn',
        'transformers',
        'torch',
        'xlrd',
        'tqdm',
        'jieaba',
        'importlib.resources',
        'psutil',
        'matplotlib'
        'seaborn',
        'gensim',
        'bertopic',
        'nltk',
        'openai'
    ],
    python_requires='>=3.6',
)