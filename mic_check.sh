python - <<'PY'
import sounddevice as sd, numpy as np, time
dev = 2  # MacBook Pro Microphone
with sd.InputStream(device=dev, samplerate=16000, channels=1, dtype='float32') as s:
    for _ in range(10):
        data, _ = s.read(16000//4)
        print("RMS", float(np.sqrt((data**2).mean())))
        time.sleep(0.25)
PY
