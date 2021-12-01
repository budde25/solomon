mod consts;
mod galois;
mod inversion_tree;
mod matrix;
mod rs;

#[cfg(test)]
mod tests {
    use crate::rs::Encoder;
    use crate::rs::ReedSolo;

    #[test]
    fn simple() {
        let encoder = ReedSolo::new(4, 2).unwrap();
        let data = b"Hello World!";
        let mut shards = encoder.split(data).unwrap();
        encoder.encode(&mut shards).unwrap();
        assert_eq!(encoder.verify(&shards).unwrap(), true);
        shards[1] = Vec::new();
        assert_eq!(encoder.verify(&shards).unwrap(), false);
        encoder.reconstuct(&mut shards).unwrap();
        dbg!(&shards);
        let res = encoder.join(&shards, 12).unwrap();
        assert_eq!("Hello World!".to_string(), String::from_utf8(res).unwrap());
    }
}
