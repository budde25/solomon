use crate::{
    galois,
    matrix::{Matrix, MatrixError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderError {
    TooFewShards,
    ShortData,

    ReedSolo(ReedSoloError),
}

impl From<ReedSoloError> for EncoderError {
    fn from(e: ReedSoloError) -> Self {
        Self::ReedSolo(e)
    }
}

pub trait Encoder {
    fn encode(&self, shards: &mut Vec<Vec<u8>>) -> Result<(), EncoderError>;
    fn reconstuct(&self, shards: &mut Vec<Vec<u8>>) -> Result<(), EncoderError>;
    fn verify(&self, shards: &[Vec<u8>]) -> Result<bool, EncoderError>;
    fn split(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncoderError>;
    fn join(&self, shards: &[Vec<u8>], out_size: usize) -> Result<Vec<u8>, EncoderError>;
}

fn build_matrix(data_shards: usize, total_shards: usize) -> Result<Matrix, MatrixError> {
    let vm = Matrix::vandermonde(total_shards, data_shards)?;
    let top = vm.sub_matrix(0, 0, data_shards, data_shards)?;
    let top_inv = top.inverse()?;
    Ok(vm * top_inv)
}

#[derive(Debug, Clone)]
pub struct ReedSolo {
    data_shards: usize,
    parity_shards: usize,
    shards: usize,
    matrix: Matrix,
    parity: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReedSoloError {
    InvalidShardNumber,
    InvalidShardSize,
    ShardNoData,
}

impl ReedSolo {
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, ReedSoloError> {
        if data_shards == 0 || parity_shards == 0 {
            //TODO allow zero parity shards
            return Err(ReedSoloError::InvalidShardNumber);
        }
        if data_shards + parity_shards > 256 {
            return Err(ReedSoloError::InvalidShardNumber);
        }

        let matrix = build_matrix(data_shards, data_shards + parity_shards).unwrap();
        let mut parity = Vec::with_capacity(parity_shards);
        for i in 0..parity_shards {
            let range = &matrix[data_shards + i];
            parity.push(range.to_vec());
        }

        Ok(Self {
            parity_shards,
            data_shards,
            shards: parity_shards + data_shards,
            matrix,
            parity,
        })
    }

    fn code_some_shards(
        &self,
        matrix_rows: &[&[u8]],
        inputs: Vec<Vec<u8>>,
        outputs: &mut Vec<Vec<u8>>,
        output_count: usize,
    ) {
        if outputs.is_empty() {
            return;
        }

        for _ in 0..inputs[0].len() {
            for (c, input) in inputs.iter().enumerate().take(self.data_shards) {
                for row in 0..output_count {
                    let out = outputs[row].as_mut();
                    if c == 0 {
                        galois::gal_mul_slice(matrix_rows[row][c], input.as_slice(), out)
                    } else {
                        galois::gal_mul_slice_xor(matrix_rows[row][c], input.as_slice(), out)
                    }
                }
            }
        }
    }

    fn inner_reconstuct(
        &self,
        shards: &mut Vec<Vec<u8>>,
        data_only: bool,
    ) -> Result<(), EncoderError> {
        if shards.len() != self.shards {
            return Err(EncoderError::TooFewShards);
        }

        let shard_size = {
            let no_mut: Vec<&[u8]> = shards.iter().map(|v| v.as_ref()).collect();
            Self::check_shards(no_mut.as_slice())?;
            Self::shard_size(no_mut.as_slice())
        };

        let mut number_present = 0;
        let mut data_present = 0;
        for (i, shard) in shards.iter().enumerate().take(self.shards) {
            if !shard.is_empty() {
                number_present += 1;
                if i < self.data_shards {
                    data_present += 1;
                }
            }
        }

        if number_present == self.shards || data_only && data_present == self.data_shards {
            return Ok(());
        }

        if number_present < self.data_shards {
            return Err(EncoderError::TooFewShards);
        }

        let mut sub_shards: Vec<Vec<u8>> = vec![Vec::new(); self.data_shards];
        let mut valid_indices: Vec<usize> = vec![0; self.data_shards];
        let mut invalid_indices: Vec<usize> = Vec::new();
        let mut sub_matrix_row = 0;
        for (i, shard) in shards.iter().enumerate().take(self.shards) {
            if sub_matrix_row >= self.data_shards {
                break;
            }

            if !shard.is_empty() {
                sub_shards[sub_matrix_row] = shard.clone();
                valid_indices[sub_matrix_row] = i;
                sub_matrix_row += 1;
            } else {
                invalid_indices.push(i);
            }
        }

        let data_decode_matrix = {
            let mut sub_matrix = Matrix::new(self.data_shards, self.data_shards).unwrap();
            for r in 0..valid_indices.len() {
                let v = valid_indices[r];
                for c in 0..self.data_shards {
                    sub_matrix[r][c] = self.matrix[v][c];
                }
            }
            sub_matrix.inverse().unwrap()
        };

        let mut outputs = Vec::new();
        let mut matrix_rows: Vec<Vec<u8>> = Vec::new();
        for _ in 0..self.parity_shards {
            outputs.push(Vec::new());
            matrix_rows.push(Vec::new());
        }
        let mut output_count = 0;

        for i_shard in 0..self.data_shards {
            if shards[i_shard].is_empty() {
                outputs[output_count] = shards[i_shard].clone();

                if outputs[output_count].is_empty() {
                    for _ in 0..shard_size {
                        outputs[output_count].push(0);
                    }
                }

                matrix_rows[output_count] = data_decode_matrix[i_shard].to_vec();
                output_count += 1;
            }
        }

        let mut output = outputs[0..output_count].to_vec();
        let matrix_rows: Vec<&[u8]> = matrix_rows.iter().map(|f| f.as_slice()).collect();

        self.code_some_shards(&matrix_rows, sub_shards, &mut output, output_count);

        let mut counter = 0;
        for shard in shards.iter_mut().take(self.data_shards) {
            if shard.is_empty() {
                *shard = output[counter].clone();
                counter += 1;
            }
        }

        if data_only {
            return Ok(());
        }

        let mut matrix_rows = matrix_rows;
        let mut output_count = 0;
        for i_shard in self.data_shards..self.shards {
            if shards[i_shard].is_empty() {
                outputs[output_count] = shards[i_shard].clone();

                if outputs[output_count].is_empty() {
                    for _ in 0..shard_size {
                        outputs[output_count].push(0);
                    }
                }

                matrix_rows[output_count] = self.parity[i_shard - self.data_shards].as_slice();
                output_count += 1;
            }
        }
        let mut output = outputs[0..output_count].to_vec();
        let inputs = shards[0..self.data_shards].to_vec();
        self.code_some_shards(&matrix_rows, inputs, &mut output, output_count);

        let mut counter = 0;
        for shard in shards
            .iter_mut()
            .skip(self.data_shards)
            .take(self.parity_shards)
        {
            if shard.is_empty() {
                *shard = output[counter].clone();
                counter += 1;
            }
        }

        Ok(())
    }

    fn check_some_shards(
        &self,
        matrix_rows: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        to_check: Vec<Vec<u8>>,
        output_count: usize,
        byte_count: usize,
    ) -> bool {
        if to_check.is_empty() {
            return true;
        }
        let mut outputs = Vec::with_capacity(to_check.len());
        for i in 0..to_check.len() {
            outputs.push(Vec::with_capacity(byte_count));
            for _ in 0..byte_count {
                outputs[i].push(0)
            }
        }

        let mat_rows: Vec<&[u8]> = matrix_rows.iter().map(|v| v.as_slice()).collect();
        self.code_some_shards(mat_rows.as_slice(), inputs, &mut outputs, output_count);

        for i in 0..outputs.len() {
            if outputs[i] != to_check[i] {
                return false;
            }
        }

        true
    }

    fn check_shards(shards: &[&[u8]]) -> Result<(), ReedSoloError> {
        let size = Self::shard_size(shards);
        if size == 0 {
            return Err(ReedSoloError::ShardNoData);
        }

        for shard in shards {
            if shard.len() != size && !shard.is_empty() {
                return Err(ReedSoloError::InvalidShardSize);
            }
        }

        Ok(())
    }

    fn shard_size(shards: &[&[u8]]) -> usize {
        for shard in shards {
            if !shard.is_empty() {
                return shard.len();
            }
        }
        0
    }
}

impl Encoder for ReedSolo {
    fn encode(&self, shards: &mut Vec<Vec<u8>>) -> Result<(), EncoderError> {
        if shards.len() != self.shards {
            return Err(EncoderError::TooFewShards);
        }

        {
            let no_mut: Vec<&[u8]> = shards.iter().map(|v| v.as_ref()).collect();
            Self::check_shards(no_mut.as_slice())?;
        }

        let inputs = shards[0..self.data_shards].to_vec();
        let mut output: Vec<Vec<u8>> = shards[self.data_shards..].to_vec();
        let matrix_rows: Vec<&[u8]> = self.parity.iter().map(|v| v.as_slice()).collect();

        self.code_some_shards(
            matrix_rows.as_slice(),
            inputs,
            &mut output,
            self.parity_shards,
        );

        for i in self.data_shards..shards.len() {
            shards[i].copy_from_slice(output[i - self.data_shards].as_slice());
        }

        Ok(())
    }

    fn reconstuct(&self, shards: &mut Vec<Vec<u8>>) -> Result<(), EncoderError> {
        self.inner_reconstuct(shards, false)
    }

    fn verify(&self, shards: &[Vec<u8>]) -> Result<bool, EncoderError> {
        if shards.len() != self.shards {
            return Err(EncoderError::TooFewShards);
        }

        {
            let no_mut: Vec<&[u8]> = shards.iter().map(|v| v.as_ref()).collect();
            Self::check_shards(no_mut.as_slice())?;
        }

        let to_check = shards[self.data_shards..].to_vec();

        Ok(self.check_some_shards(
            self.parity.clone(),
            shards[0..self.data_shards].to_vec(),
            to_check,
            self.parity_shards,
            shards[0].len(),
        ))
    }

    fn split(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncoderError> {
        if data.is_empty() {
            return Err(EncoderError::ShortData);
        }

        let mut overall = Vec::with_capacity(self.shards);
        let mut mutable = data.to_vec();
        let per_shard = (data.len() + self.data_shards - 1) / self.data_shards;
        for _ in 1..self.data_shards {
            let res = mutable.split_at(data.len() / self.data_shards);
            overall.push(res.0.to_vec());
            mutable = res.1.to_vec();
        }
        overall.push(mutable);

        for _ in 0..self.parity_shards {
            let padding = vec![0; per_shard];
            overall.push(padding);
        }

        Ok(overall)
    }

    fn join(&self, shards: &[Vec<u8>], out_size: usize) -> Result<Vec<u8>, EncoderError> {
        if shards.len() < self.data_shards {
            return Err(EncoderError::TooFewShards);
        }
        let new_shards = shards[0..self.data_shards].to_vec();
        let mut size = 0;

        for new_shard in &new_shards {
            size += new_shard.len();

            if size >= out_size {
                break;
            }
        }

        if size < out_size {
            return Err(EncoderError::ShortData);
        }

        let ret = new_shards.into_iter().flatten().collect();

        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use crate::rs::ReedSolo;
    use crate::rs::{Encoder, EncoderError};

    #[test]
    fn test_delete_one_data_simple() {
        // create an encoder object
        let encoder = ReedSolo::new(4, 2).unwrap();
        // data to encode
        let data = b"Hello World!";

        // data shards (could be written to files)
        let mut shards = encoder.split(data).unwrap();
        encoder.encode(&mut shards).unwrap();

        // should be valid
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        shards[1] = Vec::new(); // delete middle data

        // should no longer be valid
        assert_eq!(encoder.verify(&shards).unwrap(), false);

        // reconstruct the data
        encoder.reconstuct(&mut shards).unwrap();

        // should now be valid once again
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        // finally join in into a byte array
        let res = encoder.join(&shards, 12).unwrap();

        // ensure they are the same
        assert_eq!("Hello World!".to_string(), String::from_utf8(res).unwrap());
    }

    #[test]
    fn test_delete_two_data_simple() {
        // create an encoder object
        let encoder = ReedSolo::new(4, 2).unwrap();
        // data to encode
        let data = b"Hello World!";

        // data shards (could be written to files)
        let mut shards = encoder.split(data).unwrap();
        encoder.encode(&mut shards).unwrap();

        // should be valid
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        shards[1] = Vec::new(); // delete middle data
        shards[2] = Vec::new(); // delete middle data

        // should no longer be valid
        assert_eq!(encoder.verify(&shards).unwrap(), false);

        // reconstruct the data
        encoder.reconstuct(&mut shards).unwrap();

        // should now be valid once again
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        // finally join in into a byte array
        let res = encoder.join(&shards, 12).unwrap();

        // ensure they are the same
        assert_eq!("Hello World!".to_string(), String::from_utf8(res).unwrap());
    }

    #[test]
    fn test_delete_three_data_simple() {
        // create an encoder object
        let encoder = ReedSolo::new(4, 2).unwrap();
        // data to encode
        let data = b"Hello World!";

        // data shards (could be written to files)
        let mut shards = encoder.split(data).unwrap();
        encoder.encode(&mut shards).unwrap();

        // should be valid
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        shards[1] = Vec::new(); // delete middle data
        shards[2] = Vec::new(); // delete middle data
        shards[3] = Vec::new(); // delete middle data

        // should no longer be valid
        assert_eq!(encoder.verify(&shards).unwrap(), false);

        // should be impossible to reconstruct as we deleted too much data
        assert_eq!(
            encoder.reconstuct(&mut shards).unwrap_err(),
            EncoderError::TooFewShards
        );
    }

    #[test]
    fn test_delete_one_parity_simple() {
        // create an encoder object
        let encoder = ReedSolo::new(4, 2).unwrap();
        // data to encode
        let data = b"Hello World!";

        // data shards (could be written to files)
        let mut shards = encoder.split(data).unwrap();
        encoder.encode(&mut shards).unwrap();

        // should be valid
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        shards[5] = Vec::new(); // delete parity data

        // should no longer be valid
        assert_eq!(encoder.verify(&shards).unwrap(), false);

        // reconstruct the data
        encoder.reconstuct(&mut shards).unwrap();

        // should now be valid once again
        assert_eq!(encoder.verify(&shards).unwrap(), true);

        // finally join in into a byte array
        let res = encoder.join(&shards, 12).unwrap();

        // ensure they are the same
        assert_eq!("Hello World!".to_string(), String::from_utf8(res).unwrap());
    }
}
