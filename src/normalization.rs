use chrono::{Datelike, Timelike};

use crate::{mccrisk::mcc_risk, payload::FraudRequest};

pub const FEATURE_DIM: usize = 14;

const INV_MAX_AMOUNT: f32 = 1.0 / 10_000.0;
const INV_MAX_INSTALLMENTS: f32 = 1.0 / 12.0;
const INV_AMOUNT_VS_AVG_RATIO: f32 = 1.0 / 10.0;
const INV_HOUR: f32 = 1.0 / 23.0;
const INV_DOW: f32 = 1.0 / 6.0;
const INV_MAX_MINUTES: f32 = 1.0 / 1_440.0;
const INV_MAX_KM: f32 = 1.0 / 1_000.0;
const INV_MAX_TX_COUNT_24H: f32 = 1.0 / 20.0;
const INV_MAX_MERCHANT_AVG_AMOUNT: f32 = 1.0 / 10_000.0;

pub fn normalize_request(req: &FraudRequest) -> [f32; FEATURE_DIM] {
    let amount = clamp01(req.transaction.amount * INV_MAX_AMOUNT);
    let installments = clamp01(req.transaction.installments as f32 * INV_MAX_INSTALLMENTS);

    let amount_vs_avg = if req.customer.avg_amount > 0.0 {
        clamp01((req.transaction.amount / req.customer.avg_amount) * INV_AMOUNT_VS_AVG_RATIO)
    } else {
        1.0
    };

    let hour_of_day = req.transaction.requested_at.hour() as f32 * INV_HOUR;
    let day_of_week = req
        .transaction
        .requested_at
        .weekday()
        .num_days_from_monday() as f32
        * INV_DOW;

    let minutes_since_last_tx = req.last_transaction.as_ref().map_or(-1.0, |last| {
        let secs = req.transaction.requested_at.timestamp() - last.timestamp.timestamp();
        let minutes = (secs / 60).max(0) as f32;
        clamp01(minutes * INV_MAX_MINUTES)
    });

    let km_from_last_tx = req
        .last_transaction
        .as_ref()
        .map_or(-1.0, |last| clamp01(last.km_from_current * INV_MAX_KM));

    let unknown_merchant = if req
        .customer
        .known_merchants
        .iter()
        .any(|merchant| merchant == &req.merchant.id)
    {
        0.0
    } else {
        1.0
    };

    [
        amount,
        installments,
        amount_vs_avg,
        hour_of_day,
        day_of_week,
        minutes_since_last_tx,
        km_from_last_tx,
        clamp01(req.terminal.km_from_home * INV_MAX_KM),
        clamp01(req.customer.tx_count_24h as f32 * INV_MAX_TX_COUNT_24H),
        bool_feature(req.terminal.is_online),
        bool_feature(req.terminal.card_present),
        unknown_merchant,
        mcc_risk(&req.merchant.mcc),
        clamp01(req.merchant.avg_amount * INV_MAX_MERCHANT_AVG_AMOUNT),
    ]
}

#[inline(always)]
fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[inline(always)]
fn bool_feature(value: bool) -> f32 {
    value as u8 as f32
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::normalize_request;
    use crate::payload::{
        Customer, FraudRequest, LastTransaction, Merchant, Terminal, Transaction,
    };

    fn sample_request() -> FraudRequest {
        FraudRequest {
            id: "tx-1".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: Utc.with_ymd_and_hms(2026, 3, 11, 18, 45, 53).unwrap(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.2331036248,
            },
            last_transaction: None,
        }
    }

    #[test]
    fn normalizes_example_payload_in_expected_order() {
        let features = normalize_request(&sample_request());

        assert!((features[0] - 0.004112).abs() < 1e-6);
        assert!((features[1] - 0.16666667).abs() < 0.0001);
        assert!((features[2] - 0.05).abs() < 1e-6);
        assert!((features[3] - (18.0 / 23.0)).abs() < 0.0001);
        assert!((features[4] - (2.0 / 6.0)).abs() < 0.0001);
        assert_eq!(features[5], -1.0);
        assert_eq!(features[6], -1.0);
        assert!((features[7] - 0.029233104).abs() < 0.0001);
        assert!((features[8] - 0.15).abs() < 1e-6);
        assert_eq!(features[9], 0.0);
        assert_eq!(features[10], 1.0);
        assert_eq!(features[11], 0.0);
        assert!((features[12] - 0.15).abs() < 1e-6);
        assert!((features[13] - 0.006025).abs() < 1e-6);
    }

    #[test]
    fn uses_sentinel_values_when_last_transaction_is_missing() {
        let features = normalize_request(&sample_request());

        assert_eq!(features[5], -1.0);
        assert_eq!(features[6], -1.0);
    }

    #[test]
    fn clamps_values_and_handles_zero_customer_average() {
        let mut req = sample_request();
        req.transaction.amount = 20_000.0;
        req.transaction.installments = 36;
        req.customer.avg_amount = 0.0;
        req.customer.tx_count_24h = 100;
        req.merchant.avg_amount = 20_000.0;
        req.terminal.km_from_home = 3_000.0;

        let features = normalize_request(&req);

        assert_eq!(features[0], 1.0);
        assert_eq!(features[1], 1.0);
        assert_eq!(features[2], 1.0);
        assert_eq!(features[7], 1.0);
        assert_eq!(features[8], 1.0);
        assert_eq!(features[13], 1.0);
    }

    #[test]
    fn marks_unknown_merchant_and_defaults_unknown_mcc() {
        let mut req = sample_request();
        req.merchant.id = "MERC-999".into();
        req.merchant.mcc = "9999".into();

        let features = normalize_request(&req);

        assert_eq!(features[11], 1.0);
        assert_eq!(features[12], 0.5);
    }

    #[test]
    fn normalizes_last_transaction_fields_when_present() {
        let mut req = sample_request();
        req.last_transaction = Some(LastTransaction {
            timestamp: Utc.with_ymd_and_hms(2026, 3, 11, 14, 58, 35).unwrap(),
            km_from_current: 18.8626479774,
        });

        let features = normalize_request(&req);

        assert!((features[5] - (227.0 / 1_440.0)).abs() < 0.0001);
        assert!((features[6] - 0.018862648).abs() < 0.0001);
    }
}
