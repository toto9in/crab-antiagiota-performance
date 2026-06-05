pub fn quantize_int16(features: &[f32; 14]) -> [i16; 14] {
    let mut out = [0i16; 14];
    for (i, &x) in features.iter().enumerate() {
        out[i] = (x * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(quantize_int16(&[0.0f32; 14])[0], 0i16);
    }

    #[test]
    fn one_maps_to_32767() {
        let mut v = [0.0f32; 14];
        v[0] = 1.0;
        assert_eq!(quantize_int16(&v)[0], 32767i16);
    }

    #[test]
    fn sentinel_minus_one_maps_to_minus_32767() {
        let mut v = [0.0f32; 14];
        v[5] = -1.0;
        v[6] = -1.0;
        let q = quantize_int16(&v);
        assert_eq!(q[5], -32767i16);
        assert_eq!(q[6], -32767i16);
    }

    #[test]
    fn half_maps_to_16384() {
        let mut v = [0.0f32; 14];
        v[0] = 0.5;
        // 0.5 * 32767 = 16383.5, rounds half-away-from-zero to 16384.
        assert_eq!(quantize_int16(&v)[0], 16384i16);
    }

    #[test]
    fn full_example_vector() {
        let features: [f32; 14] = [
            0.038488, 0.25, 0.05, 0.869565, 0.333333, 0.225694, 0.018862, 0.013709, 0.15, 0.0, 1.0,
            0.0, 0.20, 0.029895,
        ];
        let q = quantize_int16(&features);
        assert_eq!(q[9], 0i16);
        assert_eq!(q[10], 32767i16);
        assert_eq!(q[11], 0i16);
        for val in q.iter() {
            assert!(*val >= -32767 && *val <= 32767);
        }
    }
}
