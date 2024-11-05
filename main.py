from server import ServerProcess
from cam import CamProcess
from net import NetProcess

from time import sleep

if __name__ == '__main__':
    net_proc = NetProcess()
    cam_proc = CamProcess()
    net_proc.add_q(cam_proc.out_que, cam_proc.in_que)
    server_proc = ServerProcess(cam_proc)
    net_proc.add_q(server_proc.net_out_que, server_proc.net_in_que)

    net_proc.start()
    server_proc.start()
    cam_proc.start()

    server_proc.join()
    cam_proc.join()
