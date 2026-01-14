# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# def rewrite(text):
#     prompt = f"Rewrite this text to be inclusive:\n{text}"
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=128)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def rewrite_text(text):
    prompt = (
        "Rewrite the following sentence using inclusive, neutral, and professional language. "
        "Remove age, gender, or aggressive framing. "
        "DO NOT repeat the original wording.\n\n"
     f"Original: {text}\n\n"
    "Rewritten:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.3
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
