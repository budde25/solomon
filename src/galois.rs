use crate::consts::{EXP_TABLE, LOG_TABLE, MUL_TABLE};

pub fn gal_add(a: u8, b: u8) -> u8 {
    a ^ b
}

pub fn gal_sub(a: u8, b: u8) -> u8 {
    a ^ b
}

/// gal_multiply multiplies to elements of the field.
pub fn gal_multiply(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }

    let log_a = LOG_TABLE[a as usize] as usize;
    let log_b = LOG_TABLE[b as usize] as usize;
    EXP_TABLE[log_a + log_b]
}

/// gal_divide is inverse of galMultiply.
pub fn gal_divide(a: u8, b: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    if b == 0 {
        panic!("Argument 'divisor' is 0")
    }

    let log_a = LOG_TABLE[a as usize] as usize;
    let log_b = LOG_TABLE[b as usize] as usize;
    let log_result = log_a.checked_sub(log_b).unwrap_or((log_a + 255) - log_b);
    EXP_TABLE[log_result]
}

/// Computes a**n.
///
/// The result will be the same as multiplying a times itself n times.
pub fn gal_exp(a: u8, n: usize) -> u8 {
    if n == 0 {
        return 1;
    }
    if a == 0 {
        return 0;
    }

    let log_a = LOG_TABLE[a as usize] as usize;
    let mut log_result = log_a * n;
    while log_result >= 255 {
        log_result -= 255;
    }
    EXP_TABLE[log_result]
}

pub fn gal_mul_slice(c: u8, input: &[u8], output: &mut [u8]) {
    if c == 1 {
        output.copy_from_slice(input);
        return;
    }

    // output = &mut output[0..input.len()];
    let mt = MUL_TABLE[c as usize];
    for n in 0..input.len() {
        output[n] = mt[input[n] as usize];
    }
}

pub fn gal_mul_slice_xor(c: u8, input: &[u8], output: &mut [u8]) {
    if c == 1 {
        for n in 0..input.len() {
            output[n] ^= input[n]
        }
        return;
    }

    // output = &mut output[0..input.len()];
    let mt = MUL_TABLE[c as usize];
    for n in 0..input.len() {
        output[n] ^= mt[input[n] as usize];
    }
}
