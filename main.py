from server import ServerProcess
from cam import CamProcess

from time import sleep

if __name__ == '__main__':
    cam_proc = CamProcess()
    server_proc = ServerProcess(cam_proc)

    server_proc.start()
    cam_proc.start()

    server_proc.join()
    cam_proc.join()
