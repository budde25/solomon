mod matrix;
mod rs;

#[cfg(test)]
mod tests {
    use crate::rs::reedsolo;

    #[test]
    fn it_works() {
        let reedsolo = reedsolo::new();
    }
}
