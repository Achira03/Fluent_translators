import cv2
import os
from ultralytics import YOLO

# 1. กำหนดที่อยู่ของไฟล์โมเดลตัวที่ 2 (ASL Words)
model_path = "best_words.pt"

# เช็คก่อนว่ามีไฟล์โมเดลอยู่จริงไหม
if not os.path.exists(model_path):
    print(f"❌ หาไฟล์โมเดลไม่เจอ! รบกวนเช็คว่าเอาไฟล์ '{model_path}' มาวางในโฟลเดอร์หรือยังครับ")
    exit()

# โหลดโมเดล
print("🧠 กำลังโหลดสมองกล AI (โหมดคำศัพท์/Words)...")
model = YOLO(model_path)

# 2. เปิดกล้อง (เปลี่ยนเลข 0 เป็น URL ของ DroidCam ได้ถ้าใช้มือถือ)
camera_source = 0 
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้ครับ รบกวนเช็คกล้องอีกที")
    exit()

print("🎥 กล้องเปิดเรียบร้อย! ลองทำท่าคำศัพท์ใส่กล้องได้เลยครับ")
print("🛑 กดปุ่ม 'q' บนคีย์บอร์ด เพื่อปิดโปรแกรม")

# 3. ลูปอ่านภาพและทำนายผลแบบ Real-time
while True:
    success, frame = cap.read()
    if not success:
        print("❌ สตรีมภาพจากกล้องหลุดครับ")
        break

    # ✨ สั่งพลิกภาพซ้าย-ขวา (แก้ปัญหากล้องกลับด้านเหมือนส่องกระจก)
    frame = cv2.flip(frame, 1)

    # นำภาพไปให้โมเดลทำนาย (conf=0.5 คือเอาเฉพาะความมั่นใจ 50% ขึ้นไป)
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # วาดกรอบสี่เหลี่ยมและชื่อคำศัพท์ลงบนภาพ
    annotated_frame = results[0].plot()

    # แสดงผลภาพขึ้นหน้าต่างใหม่
    cv2.imshow("Sign Language AI - Words Mode", annotated_frame)

    # รอรับคำสั่งปิดหน้าต่าง (กดตัว q เพื่อออก)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 กำลังปิดกล้อง...")
        break

# คืนทรัพยากรกล้องให้ระบบ และปิดหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()