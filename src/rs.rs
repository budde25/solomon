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
    fn encode(&self, shards: &mut [&mut [u8]]) -> Result<(), EncoderError>;
    fn verify(&self, shards: &[&[u8]]) -> Result<bool, EncoderError>;
    fn split(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncoderError>;
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

        return Ok(Self {
            parity_shards,
            data_shards,
            shards: parity_shards + data_shards,
            matrix,
            parity,
        });
    }

    fn code_some_shards(
        &self,
        matrix_rows: &[&[u8]],
        inputs: &[&[u8]],
        mut outputs: Vec<Vec<u8>>,
        output_count: usize,
    ) -> Vec<Vec<u8>> {
        if outputs.len() == 0 {
            return outputs;
        }

        //let mut start = 0;
        //let end = outputs[0].len();

        for _ in 0..inputs[0].len() {
            for c in 0..self.data_shards {
                let input = inputs[c];
                for row in 0..output_count {
                    let out = outputs[row].as_mut();
                    if c == 0 {
                        galois::gal_mul_slice(matrix_rows[row][c], input, out)
                    } else {
                        galois::gal_mul_slice_xor(matrix_rows[row][c], input, out)
                    }
                }
            }
        }

        return outputs;
    }

    fn check_shards(shards: &[&[u8]]) -> Result<(), ReedSoloError> {
        let size = Self::shard_size(shards);
        if size == 0 {
            return Err(ReedSoloError::ShardNoData);
        }

        for shard in shards {
            if shard.len() != size {
                if shard.len() != 0 {
                    return Err(ReedSoloError::InvalidShardSize);
                }
            }
        }

        Ok(())
    }

    fn shard_size(shards: &[&[u8]]) -> usize {
        for shard in shards {
            if shard.len() != 0 {
                return shard.len();
            }
        }
        0
    }
}

impl Encoder for ReedSolo {
    fn encode(&self, shards: &mut [&mut [u8]]) -> Result<(), EncoderError> {
        if shards.len() != self.shards {
            return Err(EncoderError::TooFewShards);
        }

        {
            let no_mut: Vec<&[u8]> = shards.iter().map(|v| v.as_ref()).collect();
            Self::check_shards(no_mut.as_slice())?;
        }

        let inputs: Vec<&[u8]> = shards[0..self.data_shards]
            .iter()
            .map(|v| v.clone().as_ref())
            .collect();
        let output: Vec<Vec<u8>> = shards[self.data_shards..]
            .iter()
            .map(|v| v.to_vec())
            .collect();
        let matrix_rows: Vec<&[u8]> = self.parity.iter().map(|v| v.as_slice()).collect();

        let output = self.code_some_shards(
            matrix_rows.as_slice(),
            inputs.as_slice(),
            output,
            self.parity_shards,
        );

        for i in self.data_shards..shards.len() {
            shards[i].copy_from_slice(output[i - self.shards].as_slice());
        }

        Ok(())
    }

    fn verify(&self, shards: &[&[u8]]) -> Result<bool, EncoderError> {
        todo!()
    }

    fn split(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncoderError> {
        if data.len() == 0 {
            return Err(EncoderError::ShortData);
        }

        let per_shard = (data.len() + self.data_shards - 1) / self.data_shards;

        todo!();
    }
}
