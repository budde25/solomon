use crate::matrix::{Matrix, MatrixError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeError {
    AlreadySet,
    MatrixNotSquare,
}

#[derive(Debug, PartialEq)]
struct InversionTree {
    root: InversionNode,
}

#[derive(Debug, Clone, PartialEq)]
struct InversionNode {
    matrix: Matrix,
    children: Box<[Option<InversionNode>]>,
}

impl InversionTree {
    fn new(data_shards: usize, parity_shards: usize) -> Self {
        let identity = Matrix::new_identity(data_shards).unwrap();
        let vec: Vec<Option<InversionNode>> = vec![None; data_shards + parity_shards];
        Self {
            root: InversionNode {
                matrix: identity,
                children: vec.into_boxed_slice(),
            },
        }
    }

    fn get(&self, invalid_indices: &[usize]) -> Option<Matrix> {
        let root = self.root.clone();
        if invalid_indices.len() == 0 {
            return Some(root.matrix);
        } else {
            return root.get(invalid_indices, 0);
        }
    }

    fn insert(
        &mut self,
        invalid_indices: &[usize],
        matrix: Matrix,
        shards: usize,
    ) -> Result<(), TreeError> {
        if invalid_indices.len() == 0 {
            return Err(TreeError::AlreadySet);
        }

        if !matrix.is_square() {
            return Err(TreeError::MatrixNotSquare);
        }

        self.root.insert(invalid_indices, matrix, shards, 0);

        Ok(())
    }
}

impl InversionNode {
    fn get(&self, invalid_indices: &[usize], parent: usize) -> Option<Matrix> {
        let first_index = invalid_indices[0];
        let node = self.children[first_index - parent].clone();
        let node = match node {
            Some(i) => i,
            None => return None,
        };

        if invalid_indices.len() > 1 {
            return node.get(&invalid_indices[1..], first_index + 1);
        }

        Some(node.matrix)
    }

    fn insert(&mut self, invalid_indices: &[usize], matrix: Matrix, shards: usize, parent: usize) {
        let first_index = invalid_indices[0];
        let node = self.children[first_index - parent].clone();

        if node == None {
            let children: Vec<Option<InversionNode>> = vec![None; shards - first_index];
            let inversion_node = InversionNode {
                children: children.into_boxed_slice(),
                matrix: matrix.clone(),
            };
            self.children[first_index - parent] = Some(inversion_node);
        }

        let node = self.children[first_index - parent].clone();

        if invalid_indices.len() > 1 {
            node.unwrap()
                .insert(&invalid_indices[1..], matrix, shards, first_index + 1);
        } else {
            // todo maybe fix this
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{self, Matrix};

    #[test]
    fn test_new() {
        let tree = InversionTree::new(3, 2);

        assert_eq!(tree.root.children.len(), 5);
        assert_eq!(tree.root.matrix, Matrix::new_identity(3).unwrap());
    }

    #[test]
    fn test_get_inverted_matrix() {
        let mut tree = InversionTree::new(3, 2);

        let matrix = tree.get(&[]).unwrap();
        assert_eq!(matrix, Matrix::new_identity(3).unwrap());
        assert_eq!(tree.get(&[1]), None);
        assert_eq!(tree.get(&[1, 2]), None);

        let matrix = Matrix::new(3, 3).unwrap();
        tree.insert(&[1], matrix.clone(), 5)
            .expect("Insert should succeed");

        let cached_matrix = tree.get(&[1]).unwrap();
        assert_eq!(matrix, cached_matrix);
    }

    #[test]
    fn test_insert_inverted_matrix() {
        let mut tree = InversionTree::new(3, 2);

        let matrix = Matrix::new(3, 3).unwrap();
        tree.insert(&[1], matrix.clone(), 5)
            .expect("Failed to insert new Matrix");

        tree.insert(&[], matrix.clone(), 5)
            .expect_err("Should have failed to insert into the root node matrix");

        let matrix = Matrix::new(3, 2).unwrap();
        tree.insert(&[2], matrix.clone(), 5)
            .expect_err("msShould have failed inserting a non-square matrix");

        let matrix = Matrix::new(3, 3).unwrap();
        tree.insert(&[0, 1], matrix.clone(), 5)
            .expect("msFailed inserting new Matrix");
    }

    #[test]
    fn test_double_insert_inverted_matrix() {
        let mut tree = InversionTree::new(3, 2);

        let matrix = Matrix::new(3, 3).unwrap();
        tree.insert(&[1], matrix.clone(), 5)
            .expect("Failed to insert new Matrix");

        tree.insert(&[1], matrix.clone(), 5)
            .expect("Failed to insert new Matrix");

        let cached_matrix = tree.get(&[1]).expect("Should exist");
        assert_eq!(matrix, cached_matrix);
    }
}
