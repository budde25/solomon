mod galois;
mod matrix;
mod rs;

#[cfg(test)]
mod tests {
    use crate::rs::ReedSolo;

    #[test]
    fn it_works() {
        let reedsolo = ReedSolo::new();
    }
}
