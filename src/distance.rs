pub fn squared_distance_i16_and_f32(a: &[i16; 14], b: &[f32; 14]) -> f32 {
    let mut sum = 0.0;

    for i in 0..14 {
        let diff = a[i] as f32 - b[i];
        sum += diff * diff;
    }
    sum
}

/// Squared Euclidean distance between two int16 vectors.
/// Cast to i64 before subtracting and accumulate in u64 — an i16 difference
/// squared reaches ~4.3e9 and 14 of them overflow u32.
pub fn squared_distance_i16(a: &[i16; 14], b: &[i16; 14]) -> u64 {
    let mut sum = 0u64;

    for i in 0..14 {
        let diff = a[i] as i64 - b[i] as i64;
        sum += (diff * diff) as u64;
    }
    sum
}
