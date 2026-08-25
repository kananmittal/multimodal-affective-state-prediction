import cv2
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import time
import threading
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE SORGENTI ---
SOURCES_CONFIG = {
    #"Direttore": "Video/director.mp4",      # Cambia con 0 per webcam
    "Direttore": 0,
    "Orchestra": 1, 
    "Pubblico":  "Video/Audience.mp4"  
}

# Lista emozioni per il prompt
VALID_EMOTIONS = "joy, anger, fear, disgust, surprise, sadness, boredom, neutral"


class MultiFrameData:
    def __init__(self):
        self.latest_frames = {k: None for k in SOURCES_CONFIG.keys()}
        self.latest_emotions = {k: "Inizializzazione..." for k in SOURCES_CONFIG.keys()}
        self.lock = threading.Lock()
        self.run_analysis = True

# --- WORKER QWEN2-VL ---
def analysis_worker(data, model, processor): 
    print("🧠 Thread Qwen2-VL avviato.")
    
    # Prompt ottimizzato per Qwen (Chat Template)
    base_prompt = f" What is the general emotion? Answer with a single word from: {VALID_EMOTIONS}."

    base_prompt1 = (
        "Identify the general group emotion in this image following these specific rules: "
        #"1. Prioritize active emotions. For happiness, choose 'joy' (for smiles, laughter) over 'contentment' (for calm satisfaction). If in doubt between them, you MUST choose 'joy'. "
        "2. Distinguish 'sadness' (emotional pain, tears, downturned mouth, tense face) from 'boredom' (lack of stimulation, blank stare). If any sign of pain or distress is present, choose 'sadness'. "
        "3. Distinguish 'anger' (hostility, furrowed brows, tense jaw) from 'disgust' (revulsion, wrinkled nose, pulled-back lips). "
        "4. Use 'fear' only for reactions to a clear threat (defensive posture, shrinking back, wide eyes with tension). "
        "5. Use 'surprise' ONLY if the expression is *brief and shock-like*: wide eyes + open mouth + no signs of sustained pain, fear, anger, or disgust. "
        "   - If the same cues could indicate fear, sadness, or disgust, DO NOT choose 'surprise'. "
        "6. 'Neutral' is a LAST resort, only if there are no visible emotional cues at all. "
        f"Choose ONLY ONE or TWO emotion from the allowed list: {VALID_EMOTIONS}. "
        "Your answer MUST be in the format: 'emotions: [ em1, em2]'"
    )

    while data.run_analysis:
        frames_to_process = {}
        
        # 1. Copia sicura dei frame (Snapshot)
        with data.lock:
            for key, frame in data.latest_frames.items():
                if frame is not None:
                    frames_to_process[key] = frame.copy()
        
        if frames_to_process:
            try:
                start_global = time.time()
                results = {}

                # 2. Ciclo di inferenza sequenziale
                for key, frame_bgr in frames_to_process.items():
                    # BGR -> RGB e conversione PIL
                    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    
                    # Costruzione del messaggio stile Chat
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": base_prompt1},
                            ],
                        }
                    ]

                    # Preparazione input per Qwen
                    # apply_chat_template formatta il testo
                    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    # processor converte immagine e testo in tensori
                    inputs = processor(
                        text=[text_prompt],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # Sposta tutto su GPU
                    inputs = inputs.to("cuda")

                    # Generazione
                    # max_new_tokens=16 limita la risposta per velocità

                    generation_kwargs = {
                            "max_new_tokens": 20,
                            "do_sample": True,       # Abilita il campionamento (fondamentale)
                            "temperature": 0.6,      # 0.8 è alto. Se svalvola, scendi a 0.6. (Default era ~0)
                            "top_p": 0.9,            # Nucleus sampling: considera solo il top 90% delle probabilità                            
                        }
                    
                    output_ids = model.generate(**inputs, **generation_kwargs )
                    
                    # Decodifica della risposta (salta i token di input per avere solo la risposta)
                    generated_ids = [
                        output_ids[len(input_ids):] 
                        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                    ]
                    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    # Pulizia stringa
                    results[key] = output_text[0].strip()

                print(f"Cycle Time: {time.time() - start_global:.2f}s | Results: {results}")

                # 3. Aggiornamento dati condivisi
                with data.lock:
                    for key, val in results.items():
                        data.latest_emotions[key] = val
            
            except Exception as e:
                print(f"‼️ Errore Qwen: {e}")
                # In caso di errore OOM (Out of Memory), svuota la cache
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
        
        # Sleep per non fondere la GPU se non ci sono frame nuovi
        time.sleep(0.05) 
    print("🛑 Thread analisi terminato.")

def main_qwen_monitor():
    print("⏳ Caricamento Qwen2-VL-2B-Instruct...")
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        # Caricamento Modello
        # Usa float16 per risparmiare memoria sulla 2070
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="sdpa" 
            # Nota: Flash Attn 2 funziona solo su Ampere (RTX 30xx) o superiori di solito, 
            # sulla 2070 userà 'eager' o 'sdpa' (scaled dot product attention) automaticamente.
        )
        
        # Caricamento Processor con limiti di risoluzione
        # min_pixels e max_pixels sono CRITICI per la performance su 8GB VRAM
        # Limitiamo a circa 480p-500p massimi per l'input visivo
        processor = AutoProcessor.from_pretrained(
            model_id
        )
        
        print(f"✅ Qwen caricato su {model.device}!")

    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # --- SETUP VIDEO ---
    captures = {}
    for name, source in SOURCES_CONFIG.items():
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"✅ Sorgente '{name}' ok.")
            captures[name] = cap
        else:
            print(f"⚠️ Sorgente '{name}' non trovata.")
    
    if not captures:
        return

    data = MultiFrameData()
    
    # Avvio Thread Analisi
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, processor))
    analyzer_thread.start()

    # --- GUI MATPLOTLIB ---
    plt.ion()
    n_sources = len(captures)
    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 5))
    if n_sources == 1: axes = [axes]
    axes_map = {name: ax for name, ax in zip(captures.keys(), axes)}

    print("\n🚀 Monitoraggio Qwen avviato. Premi CTRL+C per uscire.")

    try:
        while True:
            current_frames = {}
            
            for name, cap in captures.items():
                ret, frame = cap.read()
                if ret:
                    if isinstance(SOURCES_CONFIG[name], str): 
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frames[name] = frame
                else:
                    current_frames[name] = None

            with data.lock:
                for name, frame in current_frames.items():
                    if frame is not None:
                        data.latest_frames[name] = frame
                emotions = data.latest_emotions.copy()

            # Visualizzazione
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
                else:
                    ax.text(0.5, 0.5, "NO SIGNAL", ha='center')
                ax.axis('off')

            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("Interruzione manuale...")
    finally:
        data.run_analysis = False
        analyzer_thread.join()
        for cap in captures.values():
            cap.release()
        plt.ioff()
        print("👋 Chiusura completata.")

if __name__ == "__main__":
    main_qwen_monitor()