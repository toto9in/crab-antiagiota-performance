use chrono::{Datelike, Timelike};

use crate::mcc_risk::mcc_risk;
use crate::normalization::*;
use crate::transaction::FraudScoreRequest;

pub fn vectorize(req: &FraudScoreRequest) -> [f32; 14] {
    let dt = req.transaction.requested_at;
    let hour = dt.hour() as f32 / 23.0;
    let weekday = dt.weekday().num_days_from_monday() as f32 / 6.0;

    let (minutes_since_last, km_from_last) = match &req.last_transaction {
        Some(lt) => {
            let diff = req.transaction.requested_at - lt.timestamp;
            let minutes = diff.num_minutes() as f32;
            (
                (minutes / MAX_MINUTES).clamp(0.0, 1.0),
                (lt.km_from_current / MAX_KM).clamp(0.0, 1.0),
            )
        }
        None => (-1.0, -1.0),
    };

    let unknown_merchant = if req.customer.known_merchants.contains(&req.merchant.id) {
        0.0
    } else {
        1.0
    };

    [
        (req.transaction.amount / MAX_AMOUNT).clamp(0.0, 1.0),
        (req.transaction.installments as f32 / MAX_INSTALLMENTS).clamp(0.0, 1.0),
        (req.transaction.amount / req.customer.avg_amount / AMOUNT_VS_AVG_RATIO).clamp(0.0, 1.0),
        hour,
        weekday,
        minutes_since_last,
        km_from_last,
        (req.terminal.km_from_home / MAX_KM).clamp(0.0, 1.0),
        (req.customer.tx_count_24h as f32 / MAX_TX_COUNT_24H).clamp(0.0, 1.0),
        if req.terminal.is_online { 1.0 } else { 0.0 },
        if req.terminal.card_present { 1.0 } else { 0.0 },
        unknown_merchant,
        mcc_risk(&req.merchant.mcc),
        (req.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT).clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::*;
    use chrono::DateTime;

    fn make_request(last_transaction: Option<LastTransaction>) -> FraudScoreRequest {
        FraudScoreRequest {
            id: "tx-test".into(),
            transaction: Transaction {
                amount: 384.88,
                installments: 3,
                requested_at: DateTime::parse_from_rfc3339("2026-03-11T20:23:35Z")
                    .unwrap()
                    .to_utc(),
            },
            customer: Customer {
                avg_amount: 769.76,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-009".into(), "MERC-001".into()],
            },
            merchant: Merchant {
                id: "MERC-001".into(),
                mcc: "5912".into(),
                avg_amount: 298.95,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 13.709,
            },
            last_transaction,
        }
    }

    fn make_last_tx() -> LastTransaction {
        LastTransaction {
            timestamp: DateTime::parse_from_rfc3339("2026-03-11T14:58:35Z")
                .unwrap()
                .to_utc(),
            km_from_current: 18.862,
        }
    }

    #[test]
    fn all_14_indices_typical_payload() {
        let req = make_request(Some(make_last_tx()));
        let v = vectorize(&req);
        assert_eq!(v.len(), 14);

        assert!((v[0] - 384.88 / 10_000.0).abs() < 1e-4);
        assert!((v[1] - 3.0 / 12.0).abs() < 1e-4);
        assert!((v[2] - (384.88 / 769.76 / 10.0)).abs() < 1e-4);
        assert!((v[3] - 20.0 / 23.0).abs() < 1e-4);
        assert!((v[4] - 2.0 / 6.0).abs() < 1e-4);
        assert!((v[5] - (325.0 / 1440.0)).abs() < 1e-3);
        assert!((v[6] - 18.862 / 1000.0).abs() < 1e-4);
        assert!((v[7] - 13.709 / 1000.0).abs() < 1e-4);
        assert!((v[8] - 3.0 / 20.0).abs() < 1e-4);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 0.0);
        assert_eq!(v[12], 0.20);
        assert!((v[13] - 298.95 / 10_000.0).abs() < 1e-4);
    }

    #[test]
    fn null_last_transaction_sets_sentinel() {
        let req = make_request(None);
        let v = vectorize(&req);
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
    }

    #[test]
    fn unknown_merchant_sets_flag() {
        let mut req = make_request(None);
        req.merchant.id = "MERC-UNKNOWN".into();
        assert_eq!(vectorize(&req)[11], 1.0);
    }

    #[test]
    fn known_merchant_clears_flag() {
        assert_eq!(vectorize(&make_request(None))[11], 0.0);
    }

    #[test]
    fn clamp_above_max_gives_one() {
        let mut req = make_request(None);
        req.transaction.amount = 999_999.0;
        assert_eq!(vectorize(&req)[0], 1.0);
    }

    #[test]
    fn clamp_zero_gives_zero() {
        let mut req = make_request(None);
        req.transaction.amount = 0.0;
        assert_eq!(vectorize(&req)[0], 0.0);
    }

    #[test]
    fn unknown_mcc_defaults_to_half() {
        let mut req = make_request(None);
        req.merchant.mcc = "9999".into();
        assert_eq!(vectorize(&req)[12], 0.5);
    }

    #[test]
    fn hour_of_day_boundaries() {
        let mut req = make_request(None);
        req.transaction.requested_at = DateTime::parse_from_rfc3339("2026-03-11T00:00:00Z")
            .unwrap()
            .to_utc();
        assert_eq!(vectorize(&req)[3], 0.0);

        req.transaction.requested_at = DateTime::parse_from_rfc3339("2026-03-11T23:00:00Z")
            .unwrap()
            .to_utc();
        assert_eq!(vectorize(&req)[3], 1.0);
    }

    #[test]
    fn day_of_week_boundaries() {
        let mut req = make_request(None);
        req.transaction.requested_at = DateTime::parse_from_rfc3339("2026-03-09T12:00:00Z")
            .unwrap()
            .to_utc();
        assert_eq!(vectorize(&req)[4], 0.0);

        req.transaction.requested_at = DateTime::parse_from_rfc3339("2026-03-15T12:00:00Z")
            .unwrap()
            .to_utc();
        assert_eq!(vectorize(&req)[4], 1.0);
    }
}
