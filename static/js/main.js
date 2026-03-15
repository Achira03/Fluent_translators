document.addEventListener('DOMContentLoaded', () => {
    const translateBtn = document.getElementById('translateBtn');
    const mockWordsInput = document.getElementById('mockWords');
    const jsonOutput = document.getElementById('jsonOutput');
    const loader = document.getElementById('loader');

    translateBtn.addEventListener('click', async () => {
        const words = mockWordsInput.value.trim();
        if (!words) {
            alert('กรุณาใส่คำศัพท์');
            return;
        }

        // ปิดปุ่มไว้ชั่วคราวและเปิด Loader
        jsonOutput.textContent = 'วิเคราะห์ข้อมูลภาษา...';
        jsonOutput.style.color = '#c9d1d9'; // Default Color
        loader.classList.remove('hidden');
        translateBtn.disabled = true;

        try {
            // ส่งคำขอแบบ POST ไปยัง Flask API `/api/translate`
            const response = await fetch('/api/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ words: words })
            });

            // รอรับผลลัพธ์โครงสร้าง JSON กลับมา
            const data = await response.json();
            
            // นำผลลัพธ์มาจัด Format วางลงในจอเพื่อให้ดูง่าย (ทำเยื้อง 2 space)
            jsonOutput.textContent = JSON.stringify(data, null, 2);

            // ไฮไลท์สีแยกแยะสถานะ
            if (data.error) {
                jsonOutput.style.color = '#ff7b72'; // สีแดงหากพัง หรือต่อ Ollama ไม่ได้
            } else {
                jsonOutput.style.color = '#a5d6ff'; // สีฟ้าสว่าง ถ้าทำงานปกติ
            }

        } catch (error) {
            // กรณียิง Request ไม่ผ่าน (Server ล่ม/หายไป)
            jsonOutput.textContent = 'เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์\n' + error;
            jsonOutput.style.color = '#ff7b72';
        } finally {
            // เปิดปุ่ม และซ่อน Loader
            loader.classList.add('hidden');
            translateBtn.disabled = false;
        }
    });
});
