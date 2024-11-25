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
    server_root_dir = os.path.join(BASE_DIR, "servers")

    # Update server state continuously
    while True:
        sub_dir_names = []
        for name in os.listdir(server_root_dir):
            if os.path.isdir(os.path.join(server_root_dir, name)):
                sub_dir_names.append(name)

        display_info = "All active servers are listed below:\n\n"

        for dir_name in sub_dir_names:
            sub_server_dir = f"{server_root_dir}/{dir_name}/server_list"
            server_list = os.listdir(sub_server_dir)

            display_info += f"\nIP:PORT\tSTATE\t(Service: {dir_name})\n"

            for ip_port in server_list:
                if ip_port.endswith(".flag"):
                    ip, port = ip_port.split(".flag")[0].split(":")
                    ip_info = f"{sub_server_dir}/{ip_port}"

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
