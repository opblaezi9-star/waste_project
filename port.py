import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port in ports:
    print(f"Port: {port.device}")
    print(f"Name: {port.description}")
    print(f"Hardware ID: {port.hwid}")
    print("-" * 40)