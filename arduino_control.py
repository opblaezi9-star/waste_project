import serial
import time

# Global variable to hold the serial connection
arduino = None

def init_arduino(port="COM3"):
    global arduino
    print("Connecting to Arduino...")
    try:
        arduino = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)  # Give Arduino time to reset after serial connection
        print("Arduino connected successfully!\n")
    except Exception as e:
        print(f"Warning: Could not connect to Arduino on {port}. (Running in simulation mode)")
        print(f"Error: {e}")
        arduino = None

def send_to_arduino(result):
    if arduino is None:
        print(f"[Simulation] Would send to Arduino: {result}")
        return

    if result == "biodegradable":
        arduino.write(b"biodegradable\n")
        print("Sent to Arduino: Servo -> RIGHT")

    elif result == "non biodegradable":
        arduino.write(b"non biodegradable\n")
        print("Sent to Arduino: Servo -> LEFT")

    else:
        print("Unknown AI result:", result)

def close_arduino():
    if arduino is not None:
        arduino.close()
        print("Arduino connection closed.")