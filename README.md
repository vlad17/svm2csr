# `svm2csr`: convert svmlight text files into scipy CSR representation

[![travis build](https://travis-ci.org/vlad17/svm2csr.svg?branch=master)](https://travis-ci.org/vlad17/svm2csr)


Many sparse datasets are distributed in a lightweight text format called [svmlight](http://svmlight.joachims.org/). While simple and familiar, it's terribly slow to read in python even with C++ solutions. This is a Python 3.6+ solution to loading such files by calling a parallel Rust extension which chunks files into byte blocks.

```
# benchmark dataset is kdda training set, 2.5GB flat text
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

import sklearn.datasets
%timeit %timeit sklearn.datasets.load_svmlight_file('kdda')
1min 56s ± 1.72 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# https://github.com/mblondel/svmlight-loader
%timeit svmlight_loader.load_svmlight_file('kdda')
1min 52s ± 3.11 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

import svm2csr
%timeit svm2csr.load_svmlight_file('kdda')
11.4 s ± 527 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Above micro-benchmark performed on my 8-core laptop.

# Install

```
pip install svm2csr
```

Note this package is only available for pythons, operating systems, and machine architecture targets I can build wheels for (see [Publishing](#publishing)). Right now, that makes it linux-only.

* `cp36-cp39, manylinux2010, x86_64`

# Unsupported Features

* `dtype` (currently only doubles supported)
* an svmlight ranking mode where query ids are identified with `qid`
* comments in svmlight files (start with `#`)
* empty or blank lines
* multilabel [extension](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)
* reading from compressed files
* reading from multiple files and stacking
* reading from streams
* writing SVMlight files
* `n_features` option
* graceful client `multiprocessing`
* mac and windows wheels

All of these are fixable (even stream reading with parallel bridge). Let me know if you'd like to make PR.

# Documentation

```
def load_svmlight_file(fname, zero_based="auto", min_chunk_size=(16 * 1024)):
    """
    Loads an SVMlight file into a CSR matrix.

    fname (str): the file name of the file to load.
    zero_based ("auto" or bool): whether the corresponding svmlight file uses
        zero based indexing; if false or all indices are nonzero, then
        shifts indices down uniformly by 1 for python's zero indexing.
    min_chunk_size (int): minimum chunk size in bytes per
        parallel processing task

    Returns (X, y) where X is a sparse CSR matrix and y is a numpy double array
    with length equal to the number of rows in X. Values of X are doubles.
    """
```

# Dev Info

Install maturin and pytest first.

```
pip install maturin pytest
```

Local development.

```
cargo test # test rust only
maturin develop # create py bindings for rust code
pytest # test python bindings
```

# Publishing

Maturin doesn't prepare a `setup.py` when publishing. For this reason, a source distribution doesn't make sense, as a client machine's `pip` would not know how to install this package. For this reason, only wheels are published.

A new set of wheels can be built and published for supported OSes and pythons with the following steps for a repository administrator:

1. Fetch the most recent master.
1. Bump the version in `Cargo.toml` appropriately if needed (else wheel names will clash with previous ones in pypi, though PRs should be bumping this already). Commit these changes.
1. Tag the release. `git tag -a -m "v<CURRENT VERSION>"`
1. Push to github, triggering a Travis build that tests, packages, and uploads to pypi. `git push --follow-tags`
