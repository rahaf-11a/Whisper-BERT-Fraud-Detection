import pyaudio
import numpy as np
import torch
import whisper
from transformers import pipeline # type: ignore

# تحميل نموذج Whisper للتعرف على الكلام
model_asr = whisper.load_model("medium")  # يمكن استخدام "small" أو "large" لدقة أعلى

# إعدادات الميكروفون
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz وهو معدل Whisper الافتراضي
CHUNK = 1024  # حجم كل دفعة بيانات صوتية

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("بدء التسجيل... تحدث الآن!")

# تسجيل الصوت لمدة 5 ثوانٍ
frames = []
for _ in range(0, int(RATE / CHUNK * 5)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))

print(" تم تسجيل الصوت!")

# إغلاق الميكروفون
stream.stop_stream()
stream.close()
audio.terminate()

# تحويل الصوت إلى تنسيق مناسب لـ Whisper
audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
text = model_asr.transcribe(audio_data, language="arabic")["text"]

print(" النص المستخرج:", text)

# تحميل نموذج تحليل النصوص باللغة العربية
analyzer = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-sentiment")

# تحليل النص لكشف الاحتيال
result = analyzer(text)

# تصنيف المكالمة
if "سلبي" in result[0]["label"]:  # إذا كان التحليل سلبيًا، قد يكون احتيال
    print("احتيال محتمل! ")
else:
    print("المكالمة آمنة.")
