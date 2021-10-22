use std::{fmt, io::Read, iter::FromIterator};

#[derive(PartialEq, Eq)]
pub struct Matrix {
    data: Vec<u8>,
    cols: usize,
    rows: usize,
}

impl Matrix {
    // Create a new matix of zeros
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "rows must be > 0");
        assert!(cols > 0, "cols must be > 0");
        Self {
            data: vec![0; rows * cols],
            cols,
            rows,
        }
    }

    // Create a new identity matrix
    pub fn new_identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix[i][i] = 1;
        }
        matrix
    }

    // Create a new matrix from a slice
    pub fn from_slice(rows: usize, cols: usize, slice: &[u8]) -> Self {
        assert_eq!(rows * cols, slice.len());
        Self {
            data: Vec::from_iter(slice.to_owned()),
            cols,
            rows,
        }
    }

    // Returns true if the matrix are the same size
    pub fn same_size(&self, rhs: &Matrix) -> bool {
        self.rows == rhs.rows && self.cols == rhs.cols
    }

    // Return true if the matrix is a square
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        assert!(row1 < self.rows);
        assert!(row2 < self.rows);

        todo!()
    }
}

impl std::ops::Index<usize> for Matrix {
    type Output = [u8];
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows);
        let start = index * self.cols;
        &self.data[start..(start + self.cols)]
    }
}

impl std::ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rows);
        let start = index * self.rows;
        &mut self.data[start..start + self.cols]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.into_iter()).finish()
    }
}

impl std::ops::Mul for Matrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "lhs cols do not match rhs rows");
        let mut res = Matrix::new(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    res[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        res
    }
}

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a [u8];
    type IntoIter = MatrixIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        MatrixIterator {
            matrix: self,
            index: 0,
        }
    }
}

pub struct MatrixIterator<'a> {
    matrix: &'a Matrix,
    index: usize,
}

impl<'a> Iterator for MatrixIterator<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.matrix.rows {
            return None;
        }
        let item = &self.matrix[self.index];
        self.index += 1;
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let mat = Matrix::new(2, 3);
        dbg!(mat);
    }

    #[test]
    fn test_new_identity() {
        let mat = Matrix::new_identity(3);
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[1][1], 1);
        assert_eq!(mat[2][2], 1);
        assert_eq!(mat[0][2], 0);
    }

    #[test]
    fn test_from_slice() {
        let mat = Matrix::from_slice(2, 2, &[1, 2, 3, 4]);
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[0][1], 2);
        assert_eq!(mat[1][0], 3);
        assert_eq!(mat[1][1], 4);
    }

    #[test]
    fn test_multiply() {
        let mat1 = Matrix::from_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mat2 = Matrix::from_slice(3, 2, &[7, 8, 9, 10, 11, 12]);
        let product = Matrix::from_slice(2, 2, &[58, 64, 139, 154]);
        let res = mat1 * mat2;
        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res, product);
    }
}
