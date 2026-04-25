use crate::{
    dataset::{FeatureVector, PADDED_FEATURE_DIM},
    normalization::FEATURE_DIM,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistanceKind {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

#[derive(Debug, Clone, Copy)]
pub struct DistanceEngine {
    kind: DistanceKind,
}

impl DistanceEngine {
    pub fn from_env() -> Self {
        match std::env::var("DISTANCE_IMPL").ok().as_deref() {
            Some("scalar") => Self::scalar(),
            Some("avx2") => Self::avx2().unwrap_or_else(Self::scalar),
            _ => Self::auto(),
        }
    }

    pub fn auto() -> Self {
        Self::avx2().unwrap_or_else(Self::scalar)
    }

    pub fn scalar() -> Self {
        Self {
            kind: DistanceKind::Scalar,
        }
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            DistanceKind::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            DistanceKind::Avx2 => "avx2",
        }
    }

    pub fn avx2() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Some(Self {
                    kind: DistanceKind::Avx2,
                });
            }
        }

        None
    }

    #[inline(always)]
    pub fn measure(&self, lhs: &FeatureVector, rhs: &FeatureVector) -> f32 {
        match self.kind {
            DistanceKind::Scalar => squared_euclidean_scalar(lhs, rhs),
            #[cfg(target_arch = "x86_64")]
            DistanceKind::Avx2 => unsafe { squared_euclidean_avx2(lhs, rhs) },
        }
    }
}

#[inline(always)]
pub fn squared_euclidean_scalar(lhs: &FeatureVector, rhs: &FeatureVector) -> f32 {
    let mut total = 0.0_f32;
    let mut axis = 0;
    let lhs = lhs.features();
    let rhs = rhs.features();

    while axis < FEATURE_DIM {
        let delta = lhs[axis] - rhs[axis];
        total += delta * delta;
        axis += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_euclidean_avx2(lhs: &FeatureVector, rhs: &FeatureVector) -> f32 {
    use std::arch::x86_64::{
        _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps, _mm256_add_ps,
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_load_ps, _mm256_mul_ps,
        _mm256_sub_ps,
    };

    debug_assert_eq!(PADDED_FEATURE_DIM, 16);
    let lhs_ptr = lhs.padded().as_ptr();
    let rhs_ptr = rhs.padded().as_ptr();

    let lhs_lo = unsafe { _mm256_load_ps(lhs_ptr) };
    let rhs_lo = unsafe { _mm256_load_ps(rhs_ptr) };
    let delta_lo = _mm256_sub_ps(lhs_lo, rhs_lo);
    let squared_lo = _mm256_mul_ps(delta_lo, delta_lo);

    let lhs_hi = unsafe { _mm256_load_ps(lhs_ptr.add(8)) };
    let rhs_hi = unsafe { _mm256_load_ps(rhs_ptr.add(8)) };
    let delta_hi = _mm256_sub_ps(lhs_hi, rhs_hi);
    let squared_hi = _mm256_mul_ps(delta_hi, delta_hi);

    let packed = _mm256_add_ps(squared_lo, squared_hi);
    let low128 = _mm256_castps256_ps128(packed);
    let high128 = _mm256_extractf128_ps(packed, 1);
    let sum128 = _mm_add_ps(low128, high128);
    let high64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, high64);
    let rotated = _mm_shuffle_ps(sum64, sum64, 0x55);
    let total = _mm_add_ss(sum64, rotated);

    _mm_cvtss_f32(total)
}

#[cfg(test)]
mod tests {
    use crate::dataset::FeatureVector;

    use super::{DistanceEngine, squared_euclidean_scalar};

    #[test]
    fn scalar_distance_matches_manual_sum() {
        let lhs = FeatureVector::from_features([
            0.1, 0.2, 0.3, 0.4, 0.5, -1.0, -1.0, 0.8, 0.9, 1.0, 0.0, 1.0, 0.5, 0.2,
        ]);
        let rhs = FeatureVector::from_features([
            0.0, 0.2, 0.1, 0.7, 0.1, -1.0, 0.2, 0.4, 0.1, 0.0, 1.0, 0.0, 0.3, 0.5,
        ]);
        let mut manual = 0.0_f32;
        for axis in 0..lhs.features().len() {
            let delta = lhs.features()[axis] - rhs.features()[axis];
            manual += delta * delta;
        }

        assert_eq!(squared_euclidean_scalar(&lhs, &rhs), manual);
    }

    #[test]
    fn simd_matches_scalar_when_available() {
        let Some(simd) = DistanceEngine::avx2() else {
            return;
        };
        let lhs = FeatureVector::from_features([
            0.13, 0.28, 0.33, 0.47, 0.51, -1.0, 0.05, 0.82, 0.19, 1.0, 0.0, 1.0, 0.5, 0.2,
        ]);
        let rhs = FeatureVector::from_features([
            0.11, 0.26, 0.31, 0.45, 0.43, -1.0, 0.01, 0.72, 0.29, 0.0, 1.0, 0.0, 0.3, 0.4,
        ]);

        let scalar = DistanceEngine::scalar().measure(&lhs, &rhs);
        let simd = simd.measure(&lhs, &rhs);

        assert!((simd - scalar).abs() < 1e-6);
    }
}
