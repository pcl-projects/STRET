import psutil
import random
import os
import threading
from multiprocessing import Process
import subprocess
import time
import psutil

def get_bandwidth(command):
    subprocess.call(command, shell=True)






def main(argv=None):
	command = "ping gpusrv05.cs.virginia.edu"
	p = Process(target=get_bandwidth, args=(command,))
	p.start()
	NETWORK_INTERFACE = 'enp1s0f0'
	netio1 = psutil.net_io_counters(pernic=True)
	net_usage1 = (netio1[NETWORK_INTERFACE].bytes_sent + netio1[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)

	time.sleep(5)
	# p.join()
	netio2 = psutil.net_io_counters(pernic=True)
	net_usage2 = (netio2[NETWORK_INTERFACE].bytes_sent + netio2[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)
	net_usage = net_usage2 - net_usage1

	bandwidth = net_usage / 5
	print(str(bandwidth)+" MB/s")



if __name__ == '__main__':
	main()



