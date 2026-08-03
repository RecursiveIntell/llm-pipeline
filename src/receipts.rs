//! HMAC signing and hash-chain verification for execution receipts.
use crate::types::PipelineExecutionReceiptV1;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptVerification {
    pub integrity_ok: bool,
    pub chain_ok: bool,
    pub trace_ok: bool,
    pub warnings: Vec<String>,
}

pub fn canonical_json_without_integrity(
    receipt: &PipelineExecutionReceiptV1,
) -> Result<String, serde_json::Error> {
    let mut value = serde_json::to_value(receipt)?;
    if let Some(obj) = value.as_object_mut() {
        obj.remove("integrity_tag");
        obj.remove("chain_valid");
    }
    serde_json::to_string(&value)
}

#[derive(Clone)]
pub struct ReceiptSigner {
    key: Vec<u8>,
}
impl ReceiptSigner {
    pub fn new(key: impl AsRef<[u8]>) -> Self {
        Self {
            key: key.as_ref().to_vec(),
        }
    }
    pub fn sign(
        &self,
        receipt: &mut PipelineExecutionReceiptV1,
    ) -> Result<String, serde_json::Error> {
        let canonical = canonical_json_without_integrity(receipt)?;
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .map_err(|_| serde_json::Error::io(std::io::Error::other("invalid key")))?;
        mac.update(canonical.as_bytes());
        let tag = hex::encode(mac.finalize().into_bytes());
        receipt.integrity_tag = Some(tag.clone());
        Ok(tag)
    }
}

pub struct ReceiptVerifier {
    key: Vec<u8>,
}
impl ReceiptVerifier {
    pub fn new(key: impl AsRef<[u8]>) -> Self {
        Self {
            key: key.as_ref().to_vec(),
        }
    }
    pub fn verify(
        &self,
        receipt: &mut PipelineExecutionReceiptV1,
        previous: Option<&PipelineExecutionReceiptV1>,
    ) -> ReceiptVerification {
        let mut warnings = Vec::new();
        let integrity_ok = receipt.integrity_tag.as_ref().is_some_and(|tag| {
            let canonical = canonical_json_without_integrity(receipt).unwrap_or_default();
            let mut mac = match HmacSha256::new_from_slice(&self.key) {
                Ok(mac) => mac,
                Err(_) => return false,
            };
            mac.update(canonical.as_bytes());
            hex::encode(mac.finalize().into_bytes()) == *tag
        });
        let chain_ok = match (&receipt.previous_receipt_digest, previous) {
            (None, None) => true,
            (Some(expected), Some(prev)) => {
                let bytes = serde_json::to_vec(prev).unwrap_or_default();
                let actual = hex::encode(Sha256::digest(bytes));
                actual == *expected
            }
            _ => false,
        };
        let trace_ok = receipt
            .traceparent
            .as_ref()
            .is_none_or(|tp| stack_ids::TraceCtx::from_traceparent(tp).is_ok());
        if !integrity_ok {
            warnings.push("invalid integrity tag".into());
        }
        if !chain_ok {
            warnings.push("previous receipt digest mismatch".into());
        }
        receipt.chain_valid = integrity_ok && chain_ok;
        ReceiptVerification {
            integrity_ok,
            chain_ok,
            trace_ok,
            warnings,
        }
    }
}

pub fn verify_pipeline_receipt(
    receipt: &PipelineExecutionReceiptV1,
    key: Option<&[u8; 32]>,
) -> ReceiptVerification {
    let mut copy = receipt.clone();
    key.map(|k| ReceiptVerifier::new(k).verify(&mut copy, None))
        .unwrap_or(ReceiptVerification {
            integrity_ok: false,
            chain_ok: false,
            trace_ok: true,
            warnings: vec!["missing integrity key or tag".into()],
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_test_receipt() -> PipelineExecutionReceiptV1 {
        PipelineExecutionReceiptV1 {
            receipt_version: "1".to_string(),
            crate_version: "0.2.0".to_string(),
            integrity_tag: None,
            previous_receipt_digest: None,
            traceparent: Some(
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
            ),
            tracestate: None,
            chain_valid: false,
            receipt_id: "test-001".to_string(),
            pipeline_id: "pipeline-001".to_string(),
            provider_calls: vec![],
            retry_decisions: vec![],
            budget_debits: vec![],
            response_digest: "abc123".to_string(),
            outcome: ExecutionOutcome::Success,
            recorded_time: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_sign_and_verify_roundtrip() {
        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);
        let verifier = ReceiptVerifier::new(key);

        let mut receipt = make_test_receipt();
        signer.sign(&mut receipt).unwrap();
        assert!(receipt.integrity_tag.is_some());

        let result = verifier.verify(&mut receipt, None);
        assert!(result.integrity_ok, "integrity should verify after signing");
        assert!(
            result.chain_ok,
            "chain should be ok with no previous receipt"
        );
        assert!(result.trace_ok, "traceparent should be valid");
        assert!(result.warnings.is_empty(), "no warnings expected");
        assert!(receipt.chain_valid, "chain_valid should be set to true");
    }

    #[test]
    fn test_tampered_receipt_fails_verification() {
        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);
        let verifier = ReceiptVerifier::new(key);

        let mut receipt = make_test_receipt();
        signer.sign(&mut receipt).unwrap();

        // Tamper with the receipt after signing.
        receipt.response_digest = "tampered".to_string();

        let result = verifier.verify(&mut receipt, None);
        assert!(
            !result.integrity_ok,
            "integrity should fail after tampering"
        );
        assert!(
            !receipt.chain_valid,
            "chain_valid should be false after tampering"
        );
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("invalid integrity tag")));
    }

    #[test]
    fn test_wrong_key_fails_verification() {
        let signer = ReceiptSigner::new([0u8; 32]);
        let verifier = ReceiptVerifier::new([1u8; 32]);

        let mut receipt = make_test_receipt();
        signer.sign(&mut receipt).unwrap();

        let result = verifier.verify(&mut receipt, None);
        assert!(!result.integrity_ok, "integrity should fail with wrong key");
    }

    #[test]
    fn test_chain_link_verification() {
        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);
        let verifier = ReceiptVerifier::new(key);

        let mut prev_receipt = make_test_receipt();
        prev_receipt.receipt_id = "prev-001".to_string();
        signer.sign(&mut prev_receipt).unwrap();

        let prev_digest = hex::encode(sha2::Sha256::digest(
            serde_json::to_vec(&prev_receipt).unwrap(),
        ));

        let mut receipt = make_test_receipt();
        receipt.receipt_id = "curr-001".to_string();
        receipt.previous_receipt_digest = Some(prev_digest);
        signer.sign(&mut receipt).unwrap();

        let result = verifier.verify(&mut receipt, Some(&prev_receipt));
        assert!(result.chain_ok, "chain link should verify");
        assert!(result.integrity_ok, "integrity should verify");
    }

    #[test]
    fn test_broken_chain_link_fails() {
        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);
        let verifier = ReceiptVerifier::new(key);

        let mut prev_receipt = make_test_receipt();
        prev_receipt.receipt_id = "prev-001".to_string();
        signer.sign(&mut prev_receipt).unwrap();

        let mut receipt = make_test_receipt();
        receipt.receipt_id = "curr-001".to_string();
        receipt.previous_receipt_digest = Some("wrong-digest".to_string());
        signer.sign(&mut receipt).unwrap();

        let result = verifier.verify(&mut receipt, Some(&prev_receipt));
        assert!(!result.chain_ok, "chain link should fail with wrong digest");
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("digest mismatch")));
    }

    #[test]
    fn test_traceparent_roundtrip() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let mut receipt = make_test_receipt();
        receipt.traceparent = Some(tp.to_string());

        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);
        let verifier = ReceiptVerifier::new(key);

        signer.sign(&mut receipt).unwrap();
        let result = verifier.verify(&mut receipt, None);
        assert!(result.trace_ok, "valid traceparent should pass trace check");
    }

    #[test]
    fn test_verify_pipeline_receipt_helper() {
        let key = [0u8; 32];
        let signer = ReceiptSigner::new(key);

        let mut receipt = make_test_receipt();
        signer.sign(&mut receipt).unwrap();

        let result = verify_pipeline_receipt(&receipt, Some(&key));
        assert!(result.integrity_ok, "helper should verify signed receipt");
    }

    #[test]
    fn test_verify_pipeline_receipt_without_key() {
        let receipt = make_test_receipt();
        let result = verify_pipeline_receipt(&receipt, None);
        assert!(!result.integrity_ok, "should fail without key");
        assert!(result.warnings.iter().any(|w| w.contains("missing")));
    }
}
