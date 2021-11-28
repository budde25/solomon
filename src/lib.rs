mod consts;
mod galois;
mod inversion_tree;
mod matrix;
mod rs;

#[cfg(test)]
mod tests {
    use crate::rs::ReedSolo;

    #[test]
    fn simple() {
        let encoder = ReedSolo::new(4, 2).unwrap();
        let data = b"Hello World!";
        //let shards = encoder.split(data);
        //encoder.encode(shards);
    }
}
