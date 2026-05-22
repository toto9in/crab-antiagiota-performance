pub fn mcc_risk(mcc: &str) -> f32 {
    match mcc {
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        "4511" => 0.35,
        "5311" => 0.25,
        "5999" => 0.50,
        _ => 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_mcc_returns_correct_risk() {
        assert_eq!(mcc_risk("5411"), 0.15);
        assert_eq!(mcc_risk("7995"), 0.85);
        assert_eq!(mcc_risk("5912"), 0.20);
    }

    #[test]
    fn unknown_mcc_returns_default() {
        assert_eq!(mcc_risk("9999"), 0.5);
        assert_eq!(mcc_risk(""), 0.5);
    }
}
