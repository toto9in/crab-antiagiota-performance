pub fn quantize_int8(features: &[f32; 14]) -> [i8; 14] {
    let mut out = [0i8; 14];
    for (i, &x) in features.iter().enumerate() {
        out[i] = (x * 127.0).round().clamp(-127.0, 127.0) as i8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(quantize_int8(&[0.0f32; 14])[0], 0i8);
    }

    #[test]
    fn one_maps_to_127() {
        let mut v = [0.0f32; 14];
        v[0] = 1.0;
        assert_eq!(quantize_int8(&v)[0], 127i8);
    }

    #[test]
    fn sentinel_minus_one_maps_to_minus_127() {
        let mut v = [0.0f32; 14];
        v[5] = -1.0;
        v[6] = -1.0;
        let q = quantize_int8(&v);
        assert_eq!(q[5], -127i8);
        assert_eq!(q[6], -127i8);
    }

    #[test]
    fn half_maps_to_64() {
        let mut v = [0.0f32; 14];
        v[0] = 0.5;
        assert_eq!(quantize_int8(&v)[0], 64i8);
    }

    #[test]
    fn full_example_vector() {
        let features: [f32; 14] = [
            0.038488, 0.25, 0.05, 0.869565, 0.333333,
            0.225694, 0.018862, 0.013709, 0.15,
            0.0, 1.0, 0.0, 0.20, 0.029895,
        ];
        let q = quantize_int8(&features);
        assert_eq!(q[9], 0i8);
        assert_eq!(q[10], 127i8);
        assert_eq!(q[11], 0i8);
        for val in q.iter() {
            assert!(*val >= -127 && *val <= 127);
        }
    }
}
