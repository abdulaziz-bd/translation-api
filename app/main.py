from fastapi import FastAPI, Query, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Translation API",
    description="API for translating text using NLLB model",
    version="1.0.0",
    root_path="/translation",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Language code map
supported_languages = {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "chinese": "zho_Hans",
    "arabic": "arb_Arab",
    "russian": "rus_Cyrl",
    "portuguese": "por_Latn",
    "hindi": "hin_Deva",
    "japanese": "jpn_Jpan",
    "bengali": "ben_Beng"
}

@app.get("/health", status_code=200)
def health():
    return {"status": "healthy"}

@app.get("/translate")
def translate(
    text: str = Query(..., description="Text to translate"),
    source_lang: str = Query("english", description="Source language (e.g. english, french)"),
    target_lang: str = Query("french", description="Target language (e.g. german, japanese)")
):
    if source_lang not in supported_languages or target_lang not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    src_code = supported_languages[source_lang]
    tgt_code = supported_languages[target_lang]

    # Standard tokenization
    inputs = tokenizer(text, return_tensors="pt")
    
    # Use the correct way to get the token ID for the target language
    # For NLLB, we directly convert the language code to a token ID
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code)
    )
    
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return {
        "translation": result,
        "source": source_lang,
        "target": target_lang
    }