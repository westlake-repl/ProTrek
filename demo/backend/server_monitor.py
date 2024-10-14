import os
import time
import socket


# Check whether a server is active
def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.05)

    try:
        result = sock.connect_ex((ip, port))
        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        print(e)
        return False

    finally:
        sock.close()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)
    server_dir = os.path.join(BASE_DIR, "server_list")

    # Update server state continuously
    while True:
        server_list = os.listdir(server_dir)

        display_info = "All active servers are listed below:\n\n" \
                       "IP:PORT\tSTATE\n"

        for ip_port in server_list:
            if ip_port.endswith(".flag"):
                ip, port = ip_port.split(".flag")[0].split(":")
                ip_info = f"{server_dir}/{ip_port}"

                # Remove inaccessible server
                if not check_port(ip, int(port)):
                    display_info += f"{ip}:{port}\t\033[31minaccessible\033[0m\n"

                else:
                    with open(ip_info, "r") as r:
                        state = r.read().strip()

                    if state == "idle":
                        display_info += f"{ip}:{port}\t\033[32m{state}\033[0m\n"
                    else:
                        display_info += f"{ip}:{port}\t\033[31m{state}\033[0m\n"

        os.system("clear")
        print(display_info)
        time.sleep(1)
