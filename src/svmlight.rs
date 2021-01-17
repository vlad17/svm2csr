//! Load a file as CSR end-to-end.

use std::convert::TryFrom;
use std::convert::TryInto;
use std::fs;
use std::path::Path;
use std::str;
use std::u64;

use rayon;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

use crate::delim_iter::DelimIter;
use crate::fileblocks;

#[derive(Clone, Default)]
pub struct CsrMatrix {
    pub(crate) y: Vec<f64>,
    pub(crate) data: Vec<f64>,
    pub(crate) indices: Vec<u64>,
    pub(crate) indptr: Vec<u64>,
}

/// Parse svmlight in parallel, relying on a minimum chunk size
pub fn svmlight_to_csr(fname: &Path, min_chunk_size: usize) -> CsrMatrix {
    // use a few more chunks than threads to allow rayon to load balance naturally
    let max_chunks = rayon::current_num_threads() * 4;
    let metadata = fs::metadata(fname).unwrap();
    let size: usize = metadata.len().try_into().unwrap();
    let max_chunks = max_chunks.min(size / min_chunk_size).max(1);
    let chunks = fileblocks::chunkify(&fname, max_chunks);

    let folds: Vec<_> = chunks
        .par_iter()
        .map(|chunk| {
            chunk.lines().fold(CsrMatrix::default(), |mut acc, line| {
                let start = acc.indices.len();
                let words = DelimIter::new(&line, b' ');
                let svm_line = parse(words);
                acc.y.push(svm_line.target());
                svm_line.for_each(|(feature, value)| {
                    acc.indices.push(feature);
                    acc.data.push(value);
                });
                assert!(acc.indices[start..].windows(2).all(|s| s[0] < s[1]));
                acc.indptr.push(start.try_into().unwrap());
                acc
            })
        })
        .collect();

    let nrows = folds.iter().map(|csr| csr.indptr.len()).sum();
    let ndata = folds.iter().map(|csr| csr.indices.len()).sum();
    let mut stacked = CsrMatrix {
        y: vec![0.0; nrows],
        data: vec![0.0; ndata],
        indices: vec![0; ndata],
        indptr: vec![0; nrows + 1],
    };
    stacked.indptr.truncate(nrows);
    {
        // fight the borrow checker
        let slice = CsrMatrixSlice::new(&mut stacked);
        let mut head_and_tail = slice.split(0, 0);
        let mut chunks = Vec::with_capacity(folds.len());
        for fold in &folds {
            let nrows = fold.indptr.len();
            let ndata = fold.indices.len();
            head_and_tail = head_and_tail.1.split(nrows, ndata);
            chunks.push(head_and_tail.0);
        }

        chunks
            .par_iter_mut()
            .zip(folds.par_iter())
            .for_each(|(chunk, fold)| chunk.copy_fold(fold));
    };
    assert!(stacked.y.len() == stacked.indptr.len());
    assert!(stacked.indptr.par_windows(2).all(|s| s[0] <= s[1]));
    assert!(stacked
        .indptr
        .last()
        .iter()
        .all(|&&i| i <= stacked.indices.len().try_into().unwrap()));
    stacked
        .indptr
        .push(stacked.indices.len().try_into().unwrap());

    stacked
}

/// Given a [`DelimIter`] pointing to the front of a line in a
/// simsvm file, this wrapper is a convenient iterator over
/// just the features in that line.
#[derive(Clone)]
pub struct SvmlightLineIter<'a> {
    target: &'a [u8],
    iter: DelimIter<'a>,
}

pub fn parse(mut iter: DelimIter<'_>) -> SvmlightLineIter<'_> {
    let target = iter.next().expect("target");
    SvmlightLineIter { target, iter }
}

impl<'a> Iterator for SvmlightLineIter<'a> {
    type Item = (u64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(word) = self.iter.next() {
            if word.is_empty() {
                continue;
            }
            let string = str::from_utf8(word).expect("utf-8");
            let (feature, value) = string
                .rfind(':')
                .map(|pos| (&string[..pos], &string[pos + 1..]))
                .expect("feature-value pair");
            return Some((
                feature.parse().expect("parse feature"),
                value.parse().expect("parse value"),
            ));
        }
        None
    }
}

impl<'a> SvmlightLineIter<'a> {
    pub fn target(&self) -> f64 {
        let string = str::from_utf8(self.target).expect("utf-8");
        string.parse().expect("target parse")
    }
}

struct CsrMatrixSlice<'a> {
    y: &'a mut [f64],
    data: &'a mut [f64],
    indices: &'a mut [u64],
    indptr: &'a mut [u64],
    base: u64,
}

impl<'a> CsrMatrixSlice<'a> {
    fn new(orig: &'a mut CsrMatrix) -> Self {
        Self {
            y: &mut orig.y,
            data: &mut orig.data,
            indices: &mut orig.indices,
            indptr: &mut orig.indptr,
            base: 0,
        }
    }

    fn split(self, nrows: usize, ndata: usize) -> (Self, Self) {
        let (y0, y1) = self.y.split_at_mut(nrows);
        let (data0, data1) = self.data.split_at_mut(ndata);
        let (indices0, indices1) = self.indices.split_at_mut(ndata);
        let (indptr0, indptr1) = self.indptr.split_at_mut(nrows);
        (
            Self {
                y: y0,
                data: data0,
                indices: indices0,
                indptr: indptr0,
                base: self.base,
            },
            Self {
                y: y1,
                data: data1,
                indices: indices1,
                indptr: indptr1,
                base: u64::try_from(ndata).unwrap() + self.base,
            },
        )
    }

    fn copy_fold(&mut self, from: &CsrMatrix) {
        self.y.copy_from_slice(&from.y);
        self.data.copy_from_slice(&from.data);
        self.indices.copy_from_slice(&from.indices);
        for (to, from) in self.indptr.iter_mut().zip(from.indptr.iter()) {
            *to = *from + self.base;
        }
    }
}

// tests:
//
// simple line-with-target nd line-with-two features parse line test
//
// svmlight_to_csr effectively tested by python, no need to repeat here b/c binding
// is a thin wrapper
