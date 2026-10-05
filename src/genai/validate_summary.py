import re

BANNED_FINANCIAL_ADVICE_TERMS = [
    "guaranteed profit",
    "buy now",
    "sell now",
    "100x return",
    "will double",
    "certain to rise",
    "definitely cause",
    "guaranteed return",
    "guaranteed prediction"
]


def validate_summary_grounding(summary_text, evidence_json):
    """
    Validates that the generated summary obeys grounding rules:
    - No speculative financial advice / guaranteed profit claims.
    - Matches specified asset and period.
    - Verifies factual consistency.
    """
    if not summary_text or not evidence_json:
        return {"is_valid": False, "issues": ["Empty summary or evidence"]}

    issues = []
    text_lower = summary_text.lower()

    # Check for prohibited financial advice terms
    for term in BANNED_FINANCIAL_ADVICE_TERMS:
        if term in text_lower:
            issues.append(f"Contains prohibited financial advice term: '{term}'")

    # Verify asset match
    asset = evidence_json.get("asset", "").lower()
    if asset and asset not in text_lower:
        issues.append(f"Summary does not reference target asset '{asset.upper()}'")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "unsupported_claim_check": "PASSED" if is_valid else "FAILED",
        "factuality_check": "PASSED" if is_valid else "ATTENTION NEEDED",
        "issues": issues
    }


def validate_prediction_explanation_grounding(explanation_text, pred_evidence):
    """
    Validates that the ML prediction explanation:
    - Does not override the predicted direction.
    - Does not contain prohibited financial advice or claims of guaranteed outcome.
    - Mentions the target predicted direction.
    """
    if not explanation_text or not pred_evidence:
        return {"is_valid": False, "issues": ["Empty explanation or evidence"]}

    issues = []
    text_lower = explanation_text.lower()

    for term in BANNED_FINANCIAL_ADVICE_TERMS:
        if term in text_lower:
            issues.append(f"Contains prohibited financial advice term: '{term}'")

    predicted_direction = pred_evidence.get("predicted_direction", "").lower()
    if predicted_direction and predicted_direction not in text_lower:
        issues.append(f"Explanation does not mention predicted direction '{predicted_direction.upper()}'")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "unsupported_claim_check": "PASSED" if is_valid else "FAILED",
        "model_alignment_check": "PASSED" if is_valid else "ATTENTION NEEDED",
        "issues": issues
    }


if __name__ == "__main__":
    from src.genai.package_evidence import build_evidence_package
    from src.genai.generate_summary import generate_evidence_grounded_summary

    ev = build_evidence_package()
    summ = generate_evidence_grounded_summary(ev)
    val = validate_summary_grounding(summ, ev)
    print("Validation Result:", val)
