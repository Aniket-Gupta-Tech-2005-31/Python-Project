import datetime
import time
import winsound

alarm_time = input("Enter alarm time (HH:MM:SS in 24-hour format): ")
print(f"Alarm set for {alarm_time}...")

while True:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    if now == alarm_time:
        print("Wake up! Alarm Ringing...")
        melody = [
            (1000, 400), (1200, 400), (1400, 400),
            (1000, 400), (1200, 400), (1400, 400),
            (1000, 600), (1200, 600), (1400, 600)
        ]
        for freq, dur in melody:
            winsound.Beep(freq, dur)
        break
    time.sleep(1)
