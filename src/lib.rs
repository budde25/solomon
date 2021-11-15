mod galois;
mod matrix;
mod rs;
mod inversion_tree;

#[cfg(test)]
mod tests {
    use crate::rs::ReedSolo;

    #[test]
    fn it_works() {
        let reedsolo = ReedSolo::new();
    }
}
