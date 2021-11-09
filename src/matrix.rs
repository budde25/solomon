use crate::galois::gal_multiply;

use super::galois;
use std::fmt::{self};

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum MatrixError {
    InvalidRowSize,
    InvalidColSize,
    InvalidSize,
    InvalidSliceLength,
    Singular,
}

#[derive(PartialEq, Eq)]
pub struct Matrix {
    data: Vec<u8>,
    cols: usize,
    rows: usize,
}

impl Matrix {
    // Create a new matix of zeros
    pub fn new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if rows < 1 {
            return Err(MatrixError::InvalidRowSize);
        }
        if cols < 1 {
            return Err(MatrixError::InvalidColSize);
        }
        Ok(Self {
            data: vec![0; rows * cols],
            cols,
            rows,
        })
    }

    // Create a new identity matrix
    pub fn new_identity(size: usize) -> Result<Self, MatrixError> {
        let mut matrix = Self::new(size, size)?;
        for i in 0..size {
            matrix[i][i] = 1;
        }
        Ok(matrix)
    }

    // Create a new matrix from a slice
    pub fn from_slice(rows: usize, cols: usize, slice: &[u8]) -> Result<Self, MatrixError> {
        assert_eq!(rows * cols, slice.len());
        if rows * cols != slice.len() {
            return Err(MatrixError::InvalidSliceLength);
        }
        Ok(Self {
            data: Vec::from_iter(slice.to_owned()),
            cols,
            rows,
        })
    }

    // Augment returns the concatenation of this matrix and the matrix on the right.
    pub fn augment(&self, rhs: &Matrix) -> Result<Self, MatrixError> {
        if self.rows != rhs.rows {
            return Err(MatrixError::InvalidRowSize);
        }

        let mut res = Matrix::new(self.rows, self.cols + rhs.cols)?;
        for r in 0..self.rows {
            for c in 0..self.cols {
                res[r][c] = self[r][c]
            }
            for c in 0..rhs.cols {
                res[r][self.cols + c] = rhs[r][c]
            }
        }
        Ok(res)
    }

    pub fn sub_matrix(
        &self,
        rmin: usize,
        rmax: usize,
        cmin: usize,
        cmax: usize,
    ) -> Result<Self, MatrixError> {
        let mut res = Self::new(rmax - rmin, cmax - cmin)?;
        for r in rmin..rmax {
            for c in cmin..cmax {
                res[r - rmin][c - cmin] = self[r][c];
            }
        }

        Ok(res)
    }

    /// Returns the inverse of the matrix
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::InvalidSize);
        }
        let size = self.rows;

        let work = Self::new_identity(size)?;
        let mut work = self.augment(&work)?;

        work.gaussian_elimnation()?;

        work.sub_matrix(0, size, size, size * 2)
    }

    pub fn gaussian_elimnation(&mut self) -> Result<(), MatrixError> {
        // clear out the part below the main diagonal and scale to be 1
        for r in 0..self.rows {
            if self[r][r] == 0 {
                for rb in (r + 1)..self.rows {
                    if self[rb][r] != 0 {
                        self.swap_rows(r, rb);
                        break;
                    }
                }
            }

            if self[r][r] == 0 {
                return Err(MatrixError::Singular);
            }

            if self[r][r] != 0 {
                let scale = galois::gal_divide(1, self[r][r]);
                for c in 0..self.cols {
                    self[r][c] = galois::gal_multiply(self[r][c], scale);
                }
            }

            for rb in (r + 1)..self.rows {
                if self[rb][r] != 0 {
                    let scale = self[rb][r];
                    for c in 0..self.cols {
                        self[rb][c] ^= galois::gal_multiply(self[r][c], scale);
                    }
                }
            }
        }

        for d in 0..self.rows {
            for ra in 0..d {
                if self[ra][d] != 0 {
                    let scale = self[ra][d];
                    for c in 0..self.cols {
                        self[ra][c] ^= gal_multiply(scale, self[d][c]);
                    }
                }
            }
        }

        Ok(())
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

        for i in 0..self.cols {
            let index1 = (row1 * self.cols) + i;
            let index2 = (row2 * self.cols) + i;
            let v1 = self.data[index1];
            let v2 = self.data[index2];
            self.data[index1] = v2;
            self.data[index2] = v1;
        }
    }
}

pub fn vandermonde(rows: usize, cols: usize) -> Result<Matrix, MatrixError> {
    let mut res = Matrix::new(rows, cols)?;
    for r in 0..rows {
        for c in 0..cols {
            res[r][c] = galois::gal_exp(r as u8, c)
        }
    }
    Ok(res)
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
        let start = index * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.into_iter()).finish()
    }
}

impl std::ops::Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "lhs cols do not match rhs rows");
        let mut res = Matrix::new(self.rows, rhs.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut value: u8 = 0;
                for k in 0..self.cols {
                    value ^= galois::gal_multiply(self[i][k], rhs[k][j]);
                }
                res[i][j] = value;
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
        let mat = Matrix::new(2, 3).unwrap();
        assert_eq!(mat[0][0], 0);
        assert_eq!(mat[0][1], 0);
        assert_eq!(mat[0][2], 0);
        assert_eq!(mat[1][0], 0);
        assert_eq!(mat[1][1], 0);
        assert_eq!(mat[1][2], 0);
    }

    #[test]
    fn test_new_identity() {
        let mat = Matrix::new_identity(3).unwrap();
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[1][1], 1);
        assert_eq!(mat[2][2], 1);
        assert_eq!(mat[0][2], 0);
    }

    #[test]
    fn test_from_slice() {
        let mat = Matrix::from_slice(2, 2, &[1, 2, 3, 4]).unwrap();
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[0][1], 2);
        assert_eq!(mat[1][0], 3);
        assert_eq!(mat[1][1], 4);
    }

    #[test]

    fn test_swap_rows() {
        let mut mat = Matrix::from_slice(4, 2, &[1, 2, 0, 0, 0, 0, 3, 4]).unwrap();
        mat.swap_rows(0, 3);
        assert_eq!(mat[0][0], 3);
        assert_eq!(mat[0][1], 4);
        assert_eq!(mat[3][0], 1);
        assert_eq!(mat[3][1], 2);
    }

    #[test]
    fn test_multiply() {
        let mat1 = Matrix::from_slice(2, 2, &[1, 2, 3, 4]).unwrap();
        let mat2 = Matrix::from_slice(2, 2, &[5, 6, 7, 8]).unwrap();
        let product = Matrix::from_slice(2, 2, &[11, 22, 19, 42]).unwrap();
        let res = mat1 * mat2;
        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res, product);
    }

    #[test]
    fn test_augment() {
        let mat1 = Matrix::from_slice(3, 3, &[1, 3, 2, 2, 0, 1, 5, 2, 2]).unwrap();
        let mat2 = Matrix::from_slice(3, 1, &[4, 3, 1]).unwrap();
        let res = Matrix::from_slice(3, 4, &[1, 3, 2, 4, 2, 0, 1, 3, 5, 2, 2, 1]).unwrap();
        assert_eq!(mat1.augment(&mat2).unwrap(), res);
    }

    #[test]
    fn test_sub_matix() {
        let mat = Matrix::from_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let should_be = Matrix::from_slice(2, 1, &[2, 5]).unwrap();
        let res = mat.sub_matrix(0, 2, 1, 2).unwrap();
        assert_eq!(should_be, res);
    }

    #[test]
    fn test_inverse() {
        let mat = Matrix::from_slice(3, 3, &[56, 23, 98, 3, 100, 200, 45, 201, 123]).unwrap();
        let expected =
            Matrix::from_slice(3, 3, &[175, 133, 33, 130, 13, 245, 112, 35, 126]).unwrap();
        assert_eq!(mat.inverse().unwrap(), expected);
    }
}
