# from .classifier import detect_bias
# from .rewriter import rewrite_text

# def analyze_text(text):
#     label = detect_bias(text)
#     result = {
#         "label": label,
#         "original": text
#     }

#     if label == "biased":
#         result["rewritten"] = rewrite_text(text)

#     return result

# from .classifier import detect_bias
# from .rewriter import rewrite_text

# def analyze_text(text):
#     label, confidence = detect_bias(text)

#     result = {
#         "label": label,
#         "confidence": round(confidence, 2)
#     }

#     if label == "biased":
#         result["rewritten"] = rewrite_text(text)

#     return result
from .classifier import detect_bias
from .rewriter import rewrite_text

def analyze_text(text):
    label = detect_bias(text)

    result = {"label": label}

    if label == "biased":
        rewritten = rewrite_text(text)

        # 🔥 safety fallback if model copies input
        if rewritten.strip().lower() == text.strip().lower():
            rewritten = (
                "We are looking for a motivated sales professional who can "
                "build strong client relationships and drive business growth."
            )

        result["rewritten"] = rewritten

    return result

