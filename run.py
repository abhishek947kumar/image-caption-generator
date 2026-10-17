"""
Convenience entry point to run the Image Caption Generator web application.
"""
import uvicorn
import os
import sys
import socket
import argparse

def find_free_port(start_port=8000):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return start_port

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    port = args.port or find_free_port(8001)

    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    print("=" * 60)
    print("Starting Neural Image Caption Generator Web Service...")
    print(f"Local URL: http://127.0.0.1:{port}")
    print(f"API Documentation: http://127.0.0.1:{port}/docs")
    print("=" * 60)

    # Save port for reference
    with open("server_info.json", "w") as f:
        f.write(f'{{"port": {port}, "url": "http://127.0.0.1:{port}"}}')

    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)
