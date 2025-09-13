from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Choose a smaller Granite model
MODEL_ID = "ibm-granite/granite-3.1-2B-Instruct"

print("Loading smaller Granite model:", MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",   # if you have GPU, or else CPU
    # maybe set low_cpu_mem_usage=True if transformer version supports
)

def ask_granite(prompt: str) -> str:
    try:
        output = generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        # remove prompt part from response
        generated = output[0]["generated_text"]
        # sometimes the model repeats the input; strip input from generated
        if generated.startswith(prompt):
            return generated[len(prompt):]
        else:
            return generated
    except Exception as e:
        return f"Granite error: {e}"

def analyze_text(text: str):
    simplified = ask_granite(f"Simplify this legal clause in layman-friendly language:\n{text}")
    entities = ask_granite(f"Extract key legal entities (parties, dates, obligations, amounts):\n{text}")
    doc_type = ask_granite(f"Classify the type of this document (NDA, lease, employment, service agreement, other):\n{text}")

    return {
        "original": text,
        "simplified": simplified,
        "entities": entities,
        "doc_type": doc_type,
    }

if __name__ == "__main__":
    clause = "Clause 1: Party A shall pay Party B $1000 on the first day of every month."
    result = analyze_text(clause)

    print("Original:", result["original"])
    print("Simplified:", result["simplified"])
    print("Entities:", result["entities"])
    print("Document Type:", result["doc_type"])
