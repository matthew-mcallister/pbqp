#[derive(Clone, Copy, Debug)]
pub struct Slice<'a, T> {
    storage: &'a [T],
    stride: usize,
}

impl<'a, T> Slice<'a, T> {
    pub fn len(&self) -> usize {
        self.storage.len() / self.stride
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

impl<'a, T> Iterator for Slice<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.storage.is_empty() {
            None
        } else {
            let item = &self.storage[0];
            let stride = std::cmp::min(self.stride, self.storage.len());
            self.storage = &self.storage[stride..];
            Some(item)
        }
    }
}

/// Copy-on-write matrix type.
#[derive(Clone, Debug)]
pub struct Matrix<T> {
    storage: Vec<T>,
    rows: u32,
    cols: u32,
}

impl<T> Matrix<T> {
    pub fn from_array<const M: usize, const N: usize>(slice: [[T; N]; M]) -> Self {
        let storage: Vec<T> = slice
            .into_iter()
            .flat_map(|inner| inner.into_iter())
            .collect();
        Self {
            storage,
            rows: M as u32,
            cols: N as u32,
        }
    }

    pub fn new(elems: Vec<T>, rows: u32, cols: u32) -> Self {
        assert_eq!(elems.len() as u32, rows * cols);
        Self {
            storage: elems,
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn cols(&self) -> usize {
        self.cols as usize
    }

    pub fn get_col(&self, col: u32) -> Slice<'_, T> {
        assert!(col < self.cols);
        let stride = self.cols as usize;
        let start = col as usize;
        Slice {
            storage: &self.storage[start..],
            stride,
        }
    }
}

impl<T> std::ops::Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        assert!(row < self.rows());
        &self.storage[row * self.cols as usize..(row + 1) * self.cols as usize]
    }
}
