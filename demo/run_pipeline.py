import sys
ROOT_DIR = __file__.rsplit("/", 2)[0]
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import os
import multiprocessing as mp
import time

from utils.server_tool import check_port_in_use


BASE_DIR = os.path.dirname(__file__)


def system_call(cmd: str):
    os.system(cmd)


def start_process(cmd: str):
    p = mp.Process(target=system_call, args=(cmd,))
    p.start()
    return p


def main():
    # Get the path of the Python interpreter
    python = sys.executable

    # Start the servers using multiprocessing
    processes = []

    # Start the server for embedding generation
    cmd = f"{python} {BASE_DIR}/backend/servers/embedding_generation/server.py --port 7862"
    processes.append(start_process(cmd))

    # Start the server for retrieval
    cmd = f"{python} {BASE_DIR}/backend/servers/retrieval/server.py --port 7863"
    processes.append(start_process(cmd))

    # Start the server manager in the background
    cmd = f"{python} {BASE_DIR}/backend/server_manager.py"
    processes.append(start_process(cmd))

    # Start the frontend
    cmd = f"{python} {BASE_DIR}/run.py"
    processes.append(start_process(cmd))

    # Check whether the servers are active
    print("Waiting for servers to be active...")
    while True:
        time.sleep(1)

        if sum([check_port_in_use(i) for i in [7860, 7861, 7862, 7863]]) == 4:
            print("="*50)
            print("All servers are active! You can now visit http://127.0.0.1:7860/ to start to use.")
            break

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()


