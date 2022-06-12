use std::fmt::Display;


pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
        // encoder
        TooFewShards,
        ShortData,
        // rs
        InvalidShardNumber,
        InvalidShardSize,
        ShardNoData,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::TooFewShards => write!(f, "Encoder: too few shards"),
            Error::ShortData => write!(f, "Encoder: not enough data"),
            Error::InvalidShardNumber => write!(f, "Rs: invalid shard number"),
            Error::InvalidShardSize => write!(f, "Rs: invalid shard size"),
            Error::ShardNoData => write!(f, "Rs: no data"),
        }
        
    }
}

impl std::error::Error for Error {}
