use std::rc::Rc;

/// Copy-on-write matrix type.
#[derive(Clone, Debug)]
pub struct Matrix<T> {
    storage: Rc<Vec<T>>,
    rows: u32,
    cols: u32,
    transposed: bool,
}

impl<T> Matrix<T> {
    pub fn from_array<const M: usize, const N: usize>(slice: [[T; N]; M]) -> Self {
        let storage: Vec<T> = slice
            .into_iter()
            .flat_map(|inner| inner.into_iter())
            .collect();
        Self {
            storage: Rc::new(storage),
            rows: M as u32,
            cols: N as u32,
            transposed: false,
        }
    }

    pub fn new(elems: Vec<T>, rows: u32, cols: u32) -> Self {
        assert_eq!(elems.len() as u32, rows * cols);
        Self {
            storage: Rc::new(elems),
            rows,
            cols,
            transposed: false,
        }
    }

    pub fn rows(&self) -> usize {
        if self.transposed {
            self.cols as usize
        } else {
            self.rows as usize
        }
    }

    pub fn cols(&self) -> usize {
        if self.transposed {
            self.rows as usize
        } else {
            self.cols as usize
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            storage: Rc::clone(&self.storage),
            rows: self.rows,
            cols: self.cols,
            transposed: !self.transposed,
        }
    }
}

impl<T> std::ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows() && col < self.cols());
        if self.transposed {
            &self.storage[col * self.rows as usize + row]
        } else {
            &self.storage[row * self.cols as usize + col]
        }
    }
}
