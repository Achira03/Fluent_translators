from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import ollama
import pyttsx3
import json
import threading
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ----------------------------------------------------------------------
# Section 0: Smart Word Prediction (ML Feature 1)
# ----------------------------------------------------------------------
VOCABULARY = ["HELP", "HELLO", "HUNGRY", "HAPPY", "THANK YOU", "PLEASE"]
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
# Fit vectorizer on VOCABULARY right away
vectorizer.fit(VOCABULARY)
vocab_tfidf = vectorizer.transform(VOCABULARY)

def predict_word(partial_word):
    """
    คาดเดาคำศัพท์จากตัวอักษรบางส่วน bằng TF-IDF และ Cosine Similarity
    """
    partial_upper = partial_word.upper()
    if not partial_upper:
        return partial_word
        
    input_tfidf = vectorizer.transform([partial_upper])
    similarities = cosine_similarity(input_tfidf, vocab_tfidf).flatten()
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    best_word = VOCABULARY[best_match_idx]
    
    # เพิ่มความเข้มงวด: คะแนนต้องสูง > 0.5 และความยาวคำต้องใกล้เคียงกัน (คลาดเคลื่อนไม่เกิน 2 ตัวอักษร)
    len_diff = abs(len(partial_upper) - len(best_word))
    
    if best_score > 0.5 and len_diff <= 2:
        return best_word
    return partial_word

# -------------------------------------------------i---------------------
# Section 1: การโหลด Local Model (YOLOv8 .pt)
# ----------------------------------------------------------------------
# TODO: กรุณาเปลี่ยนชื่อไฟล์ให้ตรงกับไฟล์ .pt ของคุณ
MODEL_ALPHABET_PATH = "D:\\Aj.anant\\Fluent_Translator\\train_models\\runs\\detect\\train3\\weights\\best_alp.pt"  # ไฟล์โมเดลโหมดสะกดนิ้ว A-Z
MODEL_WORDS_PATH = "best_words.pt"        # ไฟล์โมเดลโหมดคำศัพท์

model_alphabet = None
model_words = None

try:
    if os.path.exists(MODEL_ALPHABET_PATH):
        model_alphabet = YOLO(MODEL_ALPHABET_PATH)
        print(f"✅ โหลดโมเดล Alphabet ({MODEL_ALPHABET_PATH}) สำเร็จ!")
    else:
        print(f"⚠️ ไม่พบไฟล์โมเดล {MODEL_ALPHABET_PATH}")
        
    if os.path.exists(MODEL_WORDS_PATH):
        model_words = YOLO(MODEL_WORDS_PATH)
        print(f"✅ โหลดโมเดล Words ({MODEL_WORDS_PATH}) สำเร็จ!")
    else:
        print(f"⚠️ ไม่พบไฟล์โมเดล {MODEL_WORDS_PATH}")
except Exception as e:
    print(f"⚠️ คำเตือน: โหลด YOLO Model ไม่สำเร็จ: {e}")

# Global Variables สำหรับเก็บสถานะ
current_mode = 'alphabet'   # ค่า default
last_detected_word = ""     # เก็บคำล่าสุดที่กล้องจับได้
state_lock = threading.Lock() # Lock ไว้กันข้อมูลชนกันเวลารันหลาย Thread

# Variables สำหรับระบบ Auto-capture ค้าง 2 วินาที
current_holding_word = ""
holding_start_time = 0.0
auto_captured_queue = []

# ----------------------------------------------------------------------
# Section 2: ฟังก์ชันอ่านออกเสียง (TTS)
# ----------------------------------------------------------------------
def speak_text(text):
    """
    ฟังก์ชันอ่านออกเสียงภาษาไทยแบบแยก Thread ไม่ให้ UI กระตุก
    """
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
            
    threading.Thread(target=run_speech, daemon=True).start()

# ----------------------------------------------------------------------
# Section 3: ฟังก์ชันดึงภาพและส่งให้ Roboflow ตามโหมดปัจจุบัน
# ----------------------------------------------------------------------
def generate_frames():
    global last_detected_word, current_mode
    global current_holding_word, holding_start_time, auto_captured_queue
    camera = cv2.VideoCapture(0)
    
    frame_count = 0 
    # process_every_n_frames = 10 # หน่วงเวลาส่ง API เพื่อประหยัดโควต้า
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # frame = cv2.flip(frame, 1) # กลับซ้ายขวา (ปิดไว้ตามที่ผู้ใช้ต้องการ)
        frame_count += 1
        
        # เลือกโมเดลที่ต้องการรันตามสถานะ current_mode ตอนนั้น
        with state_lock:
            active_mode = current_mode
            
        model = model_alphabet if active_mode == 'alphabet' else model_words
        
        if model is not None:
            try:
                # รัน YOLO prediction บนภาพ (ตีกรอบ Confidence ต่ำสุดที่ 0.5 เหมือนไฟล์ทดสอบ)
                results = model.predict(frame, conf=0.5, verbose=False)
                
                highest_conf_label = ""
                highest_conf = 0.0
                
                # สมมติเราดึงผลจากภาพแรก (เพราะเราโยนภาพเดียว)
                if results and len(results) > 0:
                    r = results[0]
                    # ใช้ plot() ของ YOLO วาดกรอบสี่เหลี่ยมและชื่อเหมือนใน test_models.py
                    frame = r.plot()
                    
                    boxes = r.boxes
                    for box in boxes:
                        # คำนวณความแม่นยำ (Confidence) และชื่อ Label (Class Name)
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label_en = model.names[class_id] # คำศัพท์ภาษาอังกฤษที่เราเทรนไว้
                        
                        # บันทึกคำที่มีความมั่นใจสูงสุดในรอบนั้น
                        if conf > highest_conf:
                            highest_conf = conf
                            highest_conf_label = label_en
                
                # อัปเดต Global Variable เพื่อรอให้หน้าเว็บกดดึงค่าไปใช้
                with state_lock:
                    # ปรับปรุงระบบ Auto-capture (2 วินาที)
                    if highest_conf_label and highest_conf_label.lower() != "nothing":
                        last_detected_word = highest_conf_label
                        
                        # ถ้ายกคำเดิมค้างไว้
                        if highest_conf_label == current_holding_word:
                            # เกิน 2 วินาทีหรือยัง?
                            if time.time() - holding_start_time >= 2.0:
                                auto_captured_queue.append(highest_conf_label) # เก็บลงคิว
                                current_holding_word = "" # รีเซ็ตเวลาเพื่อไม่ให้จับซ้ำรัวๆ
                        else:
                            # เริ่มจับเวลาใหม่
                            current_holding_word = highest_conf_label
                            holding_start_time = time.time()
                    else:
                        # ถ้าไม่เจออะไรเลย หรือเจอคำว่า nothing ให้รีเซ็ตเวลา
                        current_holding_word = ""
                        holding_start_time = 0.0
                        
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการประมวลผล YOLO: {e}")
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------------------------------------------------
# Section 4: สร้าง Web API Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """ API สำหรับเปลี่ยนโหมดระหว่าง alphabet และ words """
    global current_mode, last_detected_word
    data = request.json
    new_mode = data.get('mode', 'alphabet')
    
    with state_lock:
        current_mode = new_mode
        last_detected_word = "" # ล้างคำเดิมทิ้งตอนเปลี่ยนโหมด
        
    print(f"🔄 เปลี่ยนโหมดการตรวจจับเป็น: {new_mode}")
    return jsonify({"status": "success", "mode": new_mode})

@app.route('/api/capture_word', methods=['GET'])
def capture_word():
    """ API ให้หน้าเว็บดูดเอาคำศัพท์ล่าสุดที่เจอไปแปะบนตะกร้า (Manual) """
    global last_detected_word
    with state_lock:
        word = last_detected_word
    return jsonify({"word": word})

@app.route('/api/check_auto_capture', methods=['GET'])
def check_auto_capture():
    """ API ให้หน้าเว็บเช็คว่ามีคำไหนค้างเกิน 2 วินาทีจนถูก Auto-capture แล้วบ้าง """
    global auto_captured_queue
    with state_lock:
        words = list(auto_captured_queue)
        auto_captured_queue.clear() # เคลียร์คิวทิ้งหลังดึงเสร็จ
    return jsonify({"words": words})

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    """ API สำหรับรับตะกร้าคำศัพท์ (Array) ไปรวมแปลภาษาไทย """
    data = request.json
    words_array = data.get('words', [])
    
    if not words_array:
        return jsonify({"error": "ไม่มีคำศัพท์ในตะกร้าให้แปล"})

    # Logic การจัดกลุ่มตัวอักษรเดี่ยวและการคาดเดาคำ (Smart Word Prediction)
    processed_words = []
    current_letters = ""
    
    for word in words_array:
        # Check if the "word" is just a single letter (length 1) and not a space
        if len(word) == 1 and word != " ":
            current_letters += word
        else:
            # If we were accumulating letters, process them first
            if current_letters:
                predicted = predict_word(current_letters)
                processed_words.append(predicted)
                current_letters = ""
            
            # Add the current full word or space
            processed_words.append(word)
            
    # Don't forget any trailing accumulated letters
    if current_letters:
        predicted = predict_word(current_letters)
        processed_words.append(predicted)

    # แปลง Array เป็น string ก่อนใส่ Prompt โดยใช้ processed_words แทน words_array
    words_json = json.dumps(processed_words, ensure_ascii=False)
    
    # System Prompt ที่ใช้ Delimiter ป้องกันการหลอน (Anti-Prompt Bleeding)
    system_prompt = """
    คุณคือผู้เชี่ยวชาญระดับสูงด้านการแปลภาษามืออเมริกัน (ASL) เป็นภาษาไทยที่สละสลวย
    
    กฎเหล็ก (Strict Rules):
    1. การจัดกลุ่มคำ: ตัวอักษรเดี่ยวที่อยู่ติดกันจะถูกรวมเป็นคำศัพท์ (เช่น "E", "Y", "E" -> "EYE")
    2. ห้ามมั่วความหมาย (No Hallucination): หากคำศัพท์รวมกันมาเป็นพรืดโดยไม่มีช่องว่าง ให้พยายามแยกเป็นคำภาษาอังกฤษที่ถูกต้องก่อน (เช่น LEGHURT ให้แยกเป็น LEG และ HURT)
    3. คำทับศัพท์ (Transliteration): หลังจากจัดกลุ่มคำแล้ว หากเป็นคำภาษาอังกฤษที่ "สามารถแปลเป็นภาษาไทยได้" (เช่น อวัยวะ, อาการ, คำนาม, กริยา ทั่วไป เช่น ARM=แขน, STOMACH=ท้อง) **ให้แปลความหมายตามปกติ ห้ามทับศัพท์เด็ดขาด!** จะทับศัพท์ก็ต่อเมื่อชุดตัวอักษรนั้นเป็น "ชื่อคน" หรือคำที่ไม่มีความหมายในภาษาอังกฤษเท่านั้น
    4. ห้ามทิ้งคำศัพท์: ต้องแปลข้อมูลจาก Input ทุกตัว และต้องนำชื่อหรือคำทับศัพท์เข้าไปใส่ในประโยค `fluent_sentence_th` ให้ครบถ้วนเสมอ ห้ามตกหล่น
    5. Chain of Thought: เขียนกระบวนการคิดวิเคราะห์ลงในฟิลด์ "thought_process" ก่อนเสมอ ว่าจะแยกคำอย่างไร คำไหนแปลความหมาย คำไหนทับศัพท์ และรวมประโยคอย่างไร
    6. การแปล: เติมคำประธาน (เช่น ฉัน, ผม) หรือคำกริยา เพื่อให้ประโยคภาษาไทยสมบูรณ์
    7. การตอบกลับ: ต้องตอบกลับเป็น JSON Object เท่านั้น ห้ามมีข้อความอื่นนอกกรอบปีกกา {} เด็ดขาด 
       JSON ต้องมีคีย์ตรงตามนี้เป๊ะๆ:
       {
           "thought_process": "การวิเคราะห์ของคุณ",
           "fluent_sentence_th": "ประโยคแปลไทย",
           "emotion_tone": "อารมณ์/ความรู้สึก"
       }
    8. ⚠️ กฎเหล็กสูงสุด: ห้ามลอกชื่อคนจากข้อความตัวอย่างด้านล่างไปตอบเด็ดขาด! ให้แปลเฉพาะคำและชื่อคนสะกดตรงตาม "Input ปัจจุบัน" ที่คุณได้รับเท่านั้น!

    --- ตัวอย่างการทำงาน (EXAMPLES ONLY - DO NOT COPY) ---
    ตัวอย่างที่ 1:
    Input: ["Help", "S", "C", "A", "R", "E"]
    Output: {
      "thought_process": "ผู้ใช้ส่งคำว่า 'Help' และสะกดคำว่า S-C-A-R-E (กลัว) ฉันรวมสองคำนี้เข้าด้วยกัน แปลว่า ช่วยด้วย ฉันกลัว พร้อมเติมประธาน 'ฉัน'",
      "fluent_sentence_th": "ช่วยด้วย! ตอนนี้ฉันรู้สึกหวาดกลัวมาก",
      "emotion_tone": "หวาดกลัว"
    }
    
    ตัวอย่างที่ 2:
    Input: ["M", "Y", "N", "A", "M", "E", "I", "S", "B", "O", "B"]
    Output: {
      "thought_process": "ตัวอักษรเรียงติดกันยาว ฉันแยกคำได้เป็น MY, NAME, IS และ BOB แต่ฐานข้อมูลชี้ว่า BOB เป็นชื่อเฉพาะ จึงอ่านออกเสียงภาษาไทยเป็น 'บ๊อบ' และแปลรวมกันทั้งหมด",
      "fluent_sentence_th": "ผมชื่อว่าบ๊อบ",
      "emotion_tone": "ทักทาย/เป็นมิตร"
    }
    -------------------------------------------------------
    """
    
    try:
        model_name = "llama3" # หรือเปลี่ยนเป็น llama3.1/openthaigpt ตามที่โหลดไว้
        
        # ตอนประกอบร่างส่งให้ Ollama
        user_message = f"Input ปัจจุบัน: {words_json}\nจงวิเคราะห์และแปลผล Input นี้เท่านั้น:"
        
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_message
            }
        ], format='json', options={'temperature': 0.1})
        
        import re
        result_str = response['message']['content']
        
        # ค้นหาข้อความที่อยู่ระหว่าง { และ } เพื่อตัดคำพูดนอกกรอบของ Ollama ทิ้ง
        json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
            try:
                result_data = json.loads(clean_json_str)
            except json.JSONDecodeError:
                result_data = {"error": "Llama 3 ตอบกลับมาในรูปแบบที่อ่านไม่ได้ (JSON parse error)"}
        else:
            result_data = {"error": "Llama 3 ไม่ได้ตอบกลับมาเป็นรูปแบบ JSON"}
        
        fluent_text_th = result_data.get("fluent_sentence_th", "")
        # เอาฟังก์ชันการพูด (pyttsx3) ออกจาก backend แล้วไปใช้ Web Speech API ใน frontend แทน
        # if fluent_text_th:
        #     speak_text(fluent_text_th)
            
        return jsonify(result_data)
        
    except Exception as e:
        error_msg = f"ข้อผิดพลาดระหว่างเชื่อมต่อ Ollama: {str(e)}"
        return jsonify({"error": error_msg})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
