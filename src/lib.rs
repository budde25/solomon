mod consts;
mod galois;
mod inversion_tree;
mod matrix;
pub mod rs;

#[cfg(test)]
mod tests {
    use crate::rs::Encoder;
    use crate::rs::ReedSolo;

    #[test]
    fn simple() {
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

        // finally join in into a byte array
        let res = encoder.join(&shards, Some(12)).unwrap();

        // ensure they are the same
        assert_eq!("Hello World!".to_string(), String::from_utf8(res).unwrap());
    }
}
