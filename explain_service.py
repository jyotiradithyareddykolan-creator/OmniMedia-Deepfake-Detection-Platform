"""
explain_service.py
~~~~~~~~~~~~~~~~~~
Generates a human-readable explanation for the detection result.
"""


def build_explanation_text(
    media_type: str,
    score: float,
    verdict: str,
    model_active: bool = False,
) -> str:
    base = f"Final forensic score: {score:.3f}. Verdict: {verdict}."

    if media_type == "audio":
        return (
            f"The audio was analysed in 3-second intervals using the fine-tuned PyTorch Acoustic Neural Network "
            f"to detect voice cloning, synthetic generation, and deepfake MFCC signatures. {base}"
        )

    if media_type == "video":
        if model_active:
            return (
                f"The video was analysed across full spatial frames using a Vision Transformer (ViT) "
                f"to detect global spatial anomalies, lighting inconsistencies, and diffusion artifacts. {base}"
            )
        return (
            f"The video was analysed frame by frame using baseline heuristics. {base}"
        )

    return f"The file was analysed with baseline forensic heuristics. {base}"
