from fastapi import FastAPI, Query, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Pipeline, AutoModelForSequenceClassification
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

# Translation model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Language detection model (using a pre-trained language identification model)
lang_model_name = "papluca/xlm-roberta-base-language-detection"
lang_tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
lang_model = AutoModelForSequenceClassification.from_pretrained(lang_model_name)

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

# Mapping from detected language codes to our supported languages
# This mapping depends on the language detection model used
xlm_roberta_to_language = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "zh": "chinese",
    "ar": "arabic",
    "ru": "russian",
    "pt": "portuguese",
    "hi": "hindi",
    "ja": "japanese",
    "bn": "bengali"
}

@app.get("/health", status_code=200)
def health():
    return {"status": "healthy"}

@app.get("/translate")
def translate(
    text: str = Query(..., description="Text to translate"),
    source_lang: str = Query(None, description="Source language (e.g. english, french). If not provided, will be auto-detected."),
    target_lang: str = Query("english", description="Target language (e.g. german, japanese)")
):
    # Auto-detect language if source_lang not provided
    detected_source = None
    if source_lang is None:
        try:
            # Use the language detection model
            inputs = lang_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = lang_model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()
            
            # Map to the language code (depends on the model - this is for papluca/xlm-roberta-base-language-detection)
            # You'll need to check the exact mapping of indices to language codes for the model you use
            detected_code = lang_model.config.id2label[predicted_label]
            
            if detected_code in xlm_roberta_to_language:
                source_lang = xlm_roberta_to_language[detected_code]
                detected_source = source_lang
            else:
                # Default to English if we can't map the detected language
                source_lang = "english"
        except Exception as e:
            # Default to English if detection fails
            source_lang = "english"
    
    if source_lang not in supported_languages or target_lang not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    src_code = supported_languages[source_lang]
    tgt_code = supported_languages[target_lang]

    # Standard tokenization
    inputs = tokenizer(text, return_tensors="pt")
    
    # Use the correct way to get the token ID for the target language
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code)
    )
    
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return {
        "translation": result,
        "source": source_lang,
        "detected_source": detected_source,
        "target": target_lang
    }