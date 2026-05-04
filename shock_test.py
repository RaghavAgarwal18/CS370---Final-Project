import RPi.GPIO as GPIO
import time

# ===== SETTINGS =====
RELAY_PIN     = 17
SHOCK_DURATION = 1.0  # seconds the TENS stays on

# ===== SETUP =====
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = off (active low board)

print("TENS Shock Test")
print("===============")
print("Make sure:")
print("  - TENS unit is ON")
print("  - Electrode pads are attached to your skin")
print("  - TENS is set to LOW intensity")
print("")

try:
    input("Press ENTER to send a test shock...")
    
    print(f"Shock ON for {SHOCK_DURATION} second(s)...")
    GPIO.output(RELAY_PIN, GPIO.LOW)   # LOW = on (active low)
    time.sleep(SHOCK_DURATION)
    
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = off (active low)
    print("Shock OFF.")
    print("")
    print("Did you feel it? If yes, your setup is working correctly!")
    print("If not, check your wiring and TENS intensity setting.")

except KeyboardInterrupt:
    print("\nTest cancelled.")

finally:
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # make sure TENS is off
    GPIO.cleanup()
    print("Done.")
