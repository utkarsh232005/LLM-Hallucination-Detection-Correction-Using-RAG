import json
import os
import textwrap

import torch
import wikipedia
from google import genai
from huggingface_hub import HfApi, create_repo, login
from scipy.special import softmax
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from torch.utils.data import DataLoader


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTED_DIR = "HalluRAG_filtered"
DATASET_PATH = os.path.join(EXTRACTED_DIR, "wikipedia_articles_used.json")
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_SAVE_PATH = "./hallucination_detector_final"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = "Shreyash03Chimote/Hallucination_Detection"


# =========================
# DATASET PREP
# =========================
def ensure_dataset(dataset_path: str) -> None:
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    if os.path.exists(dataset_path):
        return

    dummy_data = [
        {
            "title": "Solar System",
            "passage_data": [
                {
                    "reference": (
                        "The Solar System is the gravitationally bound system of the Sun "
                        "and the objects that orbit it."
                    ),
                    "llm_output": (
                        "The Solar System is the gravitationally bound system centered on "
                        "the Sun and orbiting bodies."
                    ),
                    "is_hallucinated": False,
                },
                {
                    "reference": "The Solar System formed about 4.6 billion years ago.",
                    "llm_output": "The Solar System formed 10 billion years ago.",
                    "is_hallucinated": True,
                },
            ],
        },
        {
            "title": "Leonardo da Vinci",
            "passage_data": [
                {
                    "reference": (
                        "Leonardo da Vinci was an Italian polymath of the High Renaissance."
                    ),
                    "llm_output": (
                        "Leonardo da Vinci was an Italian polymath of the High Renaissance."
                    ),
                    "is_hallucinated": False,
                },
                {
                    "reference": "He worked as a painter, engineer, and scientist.",
                    "llm_output": "He was a modern computer scientist in the 21st century.",
                    "is_hallucinated": True,
                },
            ],
        },
    ]

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Synthetic dataset created at: {dataset_path}")


def load_articles(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_training_examples(articles_data, max_examples_per_article: int = 5):
    training_data = []

    for article in articles_data:
        passages = article.get("passage_data", [])
        for passage in passages[:max_examples_per_article]:
            training_data.append(
                {
                    "reference": passage.get("reference", ""),
                    "llm_output": passage.get("llm_output", ""),
                    "is_hallucinated": passage.get("is_hallucinated", False),
                }
            )

    return training_data


# =========================
# MODEL TRAINING
# =========================
def train_model(train_examples):
    model = CrossEncoder(MODEL_NAME, num_labels=3, device=DEVICE)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    if len(train_examples) > 0:
        model.fit(
            train_dataloader=train_dataloader,
            epochs=3,
            warmup_steps=0,
            show_progress_bar=True,
        )
    else:
        print("No training examples found. Skipping training.")

    return model


# =========================
# HALLUCINATION CHECK
# =========================
def strict_hallucination_check(nli_model, llm_output: str, reference: str):
    scores = nli_model.predict([(reference, llm_output)])
    probs = softmax(scores[0])

    contradiction = probs[0]
    entailment = probs[1]

    verdict = "HALLUCINATION" if contradiction > 0.5 else "FACTUAL"
    confidence = max(contradiction, entailment)

    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "hallucination_confidence": float(contradiction),
        "factuality_confidence": float(entailment),
    }


def get_reference_from_wikipedia(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=5)
    except Exception:
        return "No reference found from Wikipedia."


def format_text(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.wrap(text, width))


# =========================
# HUGGING FACE UPLOAD
# =========================
def upload_to_huggingface(model_dir: str, repo_id: str, token: str) -> None:
    if not token:
        print("HF_TOKEN not set. Skipping upload.")
        return

    login(token=token)
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        repo_type="model",
        token=token,
    )

    print("✅ Model uploaded successfully!")
    print(f"Model URL: https://huggingface.co/{repo_id}")


# =========================
# MAIN
# =========================
def main():
    ensure_dataset(DATASET_PATH)
    articles_data = load_articles(DATASET_PATH)

    training_data = create_training_examples(articles_data)
    train_examples = [
        InputExample(
            texts=[item["llm_output"], item["reference"]],
            label=1 if not item["is_hallucinated"] else 0,
        )
        for item in training_data
        if item["llm_output"] and item["reference"]
    ]

    model = train_model(train_examples)
    model.save(MODEL_SAVE_PATH)

    # Load inference models
    nli_model = CrossEncoder(MODEL_NAME, device=DEVICE)
    _ = SentenceTransformer(SENTENCE_MODEL_NAME, device=DEVICE)  # optional similarity model

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set. Skipping Gemini request.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
    )

    llm_output = response.text or ""
    reference = get_reference_from_wikipedia(question)
    result = strict_hallucination_check(nli_model, llm_output, reference)

    print("\nLLM OUTPUT:\n")
    print(format_text(llm_output))

    print("\nREFERENCE (Wikipedia):\n")
    print(format_text(reference))

    print("\nRESULT\n")
    print("Verdict:", result["verdict"])
    print("Confidence:", result["confidence"])
    print("Hallucination Probability:", result["hallucination_confidence"])
    print("Factuality Probability:", result["factuality_confidence"])

    upload_to_huggingface(MODEL_SAVE_PATH, HF_REPO_ID, HF_TOKEN)


if __name__ == "__main__":
    main()