use std::sync::Arc;

use crate::classifier::FraudDetector;

#[derive(Clone)]
pub struct AppState {
    pub fraud_detector: Arc<FraudDetector>,
}

impl AppState {
    pub fn new(fraud_detector: Arc<FraudDetector>) -> Self {
        Self { fraud_detector }
    }
}
