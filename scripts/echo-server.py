# echo-server.py

import socket
import json
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8080  # Port to listen on (non-privileged ports are > 1023)

json_path = "./test.json"
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            with open(json_path) as file:
                payload = json.load(file)
            payload = json.dumps(payload).encode("utf-8")
            conn.sendall(payload)
            response = conn.recv(4096)
            if response:
                print(response)
                # break