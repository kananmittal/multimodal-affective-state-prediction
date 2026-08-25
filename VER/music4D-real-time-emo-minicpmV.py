import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time
import threading
import matplotlib.pyplot as plt
import os

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso trovato con il comando 'find'
#MODEL_PATH = "/home/admin1/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2_6/snapshots/SOTTO_CARTELLA_ALFANUMERICA"
MODEL_PATH = "openbmb/MiniCPM-V-4"

SOURCES_CONFIG = {
    "Direttore": "Video/director.mp4",
    "Orchestra": "Video/musician.mp4",
    "Pubblico":  "Video/Audience.mp4"
}

VALID_EMOTIONS = "joy, anger, fear, disgust, surprise, sadness, boredom, neutral"

class MultiFrameData:
    def __init__(self):
        self.latest_frames = {k: None for k in SOURCES_CONFIG.keys()}
        self.latest_emotions = {k: "Inizializzazione..." for k in SOURCES_CONFIG.keys()}
        self.lock = threading.Lock()
        self.run_analysis = True

# --- WORKER MINICPM-V ---
def analysis_worker(data, model, tokenizer): 
    print("🧠 Thread MiniCPM-V avviato.")
    
    # Prompt ottimizzato per MiniCPM
    question = f"Analyze the facial expressions and body language. Output only one word from this list: {VALID_EMOTIONS}."

    question1 = (
        "Identify the general group emotion in this image following these specific rules: "
        "1. Prioritize active emotions. "
        #"For happiness, choose 'joy' (for smiles, laughter) over 'contentment' (for calm satisfaction). If in doubt between them, you MUST choose 'joy'. "
        "2. Distinguish 'sadness' (emotional pain, tears, downturned mouth, tense face) from 'boredom' (lack of stimulation, blank stare). If any sign of pain or distress is present, choose 'sadness'. "
        "3. Distinguish 'anger' (hostility, furrowed brows, tense jaw) from 'disgust' (revulsion, wrinkled nose, pulled-back lips). "
        "4. Use 'fear' only for reactions to a clear threat (defensive posture, shrinking back, wide eyes with tension). "
        "5. Use 'surprise' ONLY if the expression is *brief and shock-like*: wide eyes + open mouth + no signs of sustained pain, fear, anger, or disgust. "
        "   - If the same cues could indicate fear, sadness, or disgust, DO NOT choose 'surprise'. "
        "6. 'Neutral' is a LAST resort, only if there are no visible emotional cues at all. "
        f"Choose ONLY ONE or TWO emotion from the allowed list: {VALID_EMOTIONS}. "
        "Your answer MUST be in the format: 'emotions: [ em1, em2]'"
    )

    question2 = (
                    f" What is the general emotion in this picture? "
                    f"Answer the question using a single word for each emotion you can find. "
                    f"Follow this example: '[ em1, em2, em3 ]'.\n"
                    f"You MUST choose only from: {VALID_EMOTIONS}."
                )

    while data.run_analysis:
        frames_to_process = {}
        
        with data.lock:
            for key, frame in data.latest_frames.items():
                if frame is not None:
                    frames_to_process[key] = frame.copy()
        
        if frames_to_process:
            try:
                start_cycle = time.time()
                results = {}

                for key, frame_bgr in frames_to_process.items():
                    # Conversione per MiniCPM
                    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert('RGB')
                    
                    # Struttura messaggio per MiniCPM-V
                    msgs = [{'role': 'user', 'content': [image, question2]}]
                    
                    # Inferenza
                    answer = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        max_new_tokens=20,
                        #sampling=False, # Più veloce
                        #temperature=0.7
                    )
                    results[key] = answer.strip()

                print(f"Cycle Time: {time.time() - start_cycle:.2f}s | Emozioni: {results}")

                with data.lock:
                    for key, val in results.items():
                        data.latest_emotions[key] = val
            
            except Exception as e:
                print(f"‼️ Errore MiniCPM: {e}")
        
        #time.sleep(0.05) 
    print("🛑 Thread analisi terminato.")

def main():
    print(f"⏳ Caricamento MiniCPM-V da: {MODEL_PATH}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Dispositivo: {device.upper()}")

    try:
        # Caricamento Modello (ottimizzato per bfloat16 se CUDA è disponibile)
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,            
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device=device)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.eval()
        print("✅ Modello caricato con successo!")
        print(f"Configurazione Attention: {model.config._attn_implementation}")
        

    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # --- SETUP VIDEO ---
    captures = {}
    for name, source in SOURCES_CONFIG.items():
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"✅ Sorgente '{name}' pronta.")
            captures[name] = cap
        else:
            print(f"⚠️ Sorgente '{name}' non trovata: {source}")
    
    if not captures:
        print("❌ Nessuna sorgente valida. Esco.")
        return

    data = MultiFrameData()
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, tokenizer))
    analyzer_thread.start()

    # --- GUI ---
    plt.ion()
    n_sources = len(captures)
    fig, axes = plt.subplots(1, n_sources, figsize=(15, 5))
    if n_sources == 1: axes = [axes]
    axes_map = {name: ax for name, ax in zip(captures.keys(), axes)}

    try:
        while True:
            current_frames = {}
            for name, cap in captures.items():
                ret, frame = cap.read()
                if ret:
                    # Loop video se è un file
                    if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frames[name] = frame
                else:
                    current_frames[name] = None

            with data.lock:
                for name, frame in current_frames.items():
                    if frame is not None:
                        data.latest_frames[name] = frame
                emotions = data.latest_emotions.copy()

            for name, ax in axes_map.items():
                ax.clear()
                frame = current_frames.get(name)
                if frame is not None:
                    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    emo = emotions.get(name, "...")

                    # Colore dinamico in base all'emozione (extra visivo)
                    bg_color = 'green' if 'joy' in emo.lower() else 'red' if 'anger' in emo.lower() else 'blue'                    
                    ax.set_title(f"{name}", fontsize=12, fontweight='bold')
                    ax.text(10, 50, emo, fontsize=14, color='white', fontweight='bold',
                            bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor='white'))
                ax.axis('off')

            plt.pause(0.01)
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("\nInterruzione richiesta...")
    finally:
        data.run_analysis = False
        analyzer_thread.join()
        for cap in captures.values():
            cap.release()
        plt.close()
        print("👋 Fine.")

if __name__ == "__main__":
    main()