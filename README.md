# SVMlight text files to scipy CSR

[![travis build](https://travis-ci.org/vlad17/svm2csr.svg?branch=master)](https://travis-ci.org/vlad17/svm2csr)
[![pypi](https://img.shields.io/pypi/v/svm2csr.svg)](https://pypi.org/project/svm2csr/)

Many sparse datasets are distributed in a lightweight text format called [svmlight](http://svmlight.joachims.org/). While simple and familiar, it's terribly slow to read in python even with C++ solutions due to serial processing. Instead, `svm2csr` loads by using a parallel Rust extension which chunks files into byte blocks, then seeks to different blocks to parse in parallel.

```
# benchmark dataset is kdda training set, 2.5GB flat text
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

import sklearn.datasets
%timeit sklearn.datasets.load_svmlight_file('kdda')
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

Note this package is only available pre-built for pythons, operating systems, and machine architecture targets I can build wheels for (see [Publishing](#publishing)). Settings other than the following need to install rust and compile from source (pip install should still work, but will compile for your platform).

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

1. Fetch the most recent master.
1. Bump the version in `Cargo.toml` appropriately if needed. Commit these changes.
1. Tag the release. `git tag -a -m "v<CURRENT VERSION>"`
1. Push to github, triggering a Travis build that tests, packages, and uploads to pypi. `git push --follow-tags`

Every master travis build attempts to publish to pypi (but may fail if a build with the same version is already present).
