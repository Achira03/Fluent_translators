import cv2
import os
from ultralytics import YOLO

# 1. กำหนดที่อยู่ของไฟล์โมเดลที่คุณเทรนเสร็จ
# ⚠️ ตรวจสอบให้แน่ใจว่า path นี้ถูกต้อง และรันไฟล์ .py ในโฟลเดอร์หลักของโปรเจคนะครับ
model_path = "D:\\Aj.anant\\Fluent_Translator\\train_models\\runs\\detect\\train3\\weights\\best_alp.pt"

# เช็คก่อนว่ามีไฟล์โมเดลอยู่จริงไหม เพื่อป้องกัน Error
if not os.path.exists(model_path):
    print(f"❌ หาไฟล์โมเดลไม่เจอ! รบกวนเช็คว่ามีไฟล์อยู่ใน '{model_path}' หรือเปล่าครับ")
    exit()

# โหลดโมเดล
print("🧠 กำลังโหลดสมองกล AI...")
model = YOLO(model_path)

# 2. เปิดกล้องเว็บแคม (เลข 0 คือกล้องตัวหลักของเครื่อง)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้ครับ รบกวนเช็คว่ามีโปรแกรมอื่นใช้กล้องอยู่ไหม")
    exit()

# ตั้งชื่อหน้าต่างและระบุคุณสมบัติให้สามารถขยายได้ (WINDOW_NORMAL)
window_name = "Sign Language AI - Test Mode"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
is_fullscreen = False

print("🎥 กล้องเปิดเรียบร้อย! ทำภาษามือใส่กล้องได้เลยครับ")
print("⌨️  กดปุ่ม 'f' เพื่อสลับโหมด เต็มจอ / หน้าต่าง")
print("🛑 กดปุ่ม 'q' บนคีย์บอร์ด เพื่อปิดโปรแกรม")

# 3. ลูปอ่านภาพและทำนายผลแบบ Real-time
while True:
    success, frame = cap.read()
    if not success:
        print("❌ สตรีมภาพจากกล้องหลุดครับ")
        break

    # นำภาพไปให้โมเดลทำนาย (conf=0.5 คือเอาเฉพาะความมั่นใจ 50% ขึ้นไป)
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # วาดกรอบสี่เหลี่ยมและชื่อคำศัพท์ลงบนภาพ
    annotated_frame = results[0].plot()

    # แสดงผลภาพขึ้นหน้าต่าง
    cv2.imshow(window_name, annotated_frame)

    # รอรับคำสั่ง (ตรวจจับการกดปุ่มทุกๆ 1 มิลลิวินาที)
    key = cv2.waitKey(1) & 0xFF
    
    # กดปุ่ม 'q' เพื่อออก
    if key == ord('q'):
        print("🛑 กำลังปิดกล้อง...")
        break
    
    # กดปุ่ม 'f' เพื่อสลับ Fullscreen
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            print("🖥️  โหมดเต็มจอ (Fullscreen On)")
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            print("🪟 โหมดหน้าต่างปกติ (Fullscreen Off)")
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# คืนทรัพยากรกล้องให้ระบบ และปิดหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()