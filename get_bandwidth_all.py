import psutil
import random
import os
import threading
from multiprocessing import Process
import subprocess
import time
import psutil

from datetime import datetime

import sys

def get_bandwidth(command):
    subprocess.call(command, shell=True)



def compute_graph():
	worker_list = []
	band_list = []
	for i in range(num_of_worker):
		file_read_band = 'resnet_my_method/my_data_resnet'+str(i)+'.csv'
	    df_band = pd.read_csv(file_read_band, header=None)
	    # print(df_itn)
	       
	    index_worker_all = pd.DataFrame(df_band.values[:,0])
	    # print(itn_time_all)

	    band_all = pd.DataFrame(df_band.values[:,1])

	    # communication_time_all = pd.DataFrame(df_cc_time.values[:,2])
	    # print(itn_time_all)

	    band_all = band_all.to_numpy()

	    # communication_time_all =communication_time_all.to_numpy()

	    index_worker_all = index_worker_all.to_numpy()

	    worker = index_worker_all[len(index_worker_all)-1]

	    band_to = band[len(band_all)-1]

	    worker_list.append(worker)
	    band_list.append(band_to)

	min_band = min(band_list)
	min_index = band_list.index(min_band)
	get_lps = worker_list[min_index]

	band_data = []
    band_data.append(datetime.now())
    band_data.append(each_stag)
    band_data.append(get_lps)
    # band_data.append(net_usage)
    # band_data.append(bandwidth)
    # common_data.append(gs)
    # common_data.append(loss_value)
    # common_data.append(duration)
    # common_data.append(accuracy)
    # common_data.append(val_accuracy)

    band_data_csv = []
    band_data_csv.append(band_data)
    import csv
    file_common = 'resnet_my/band_graph.csv' 
    with open(file_common, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(band_data_csv)


	    # workers_communication_time = communication_time_all[len(communication_time_all)-2:len(communication_time_all)]


def main(argv=None):
	local_ps_list = ['gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu']

	straggler_list = ['gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu', 'gpusrv05.cs.virginia.edu']

	for each_stag in straggler_list:
		for each_lps in local_ps_list:
			command = "ping "+str(each_lps)
			p = Process(target=get_bandwidth, args=(command,))
			p.start()
			NETWORK_INTERFACE = 'eth0'#'enp1s0f0'
			netio1 = psutil.net_io_counters(pernic=True)
			net_usage1 = (netio1[NETWORK_INTERFACE].bytes_sent + netio1[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)

			time.sleep(5)
			# p.join()
			netio2 = psutil.net_io_counters(pernic=True)
			net_usage2 = (netio2[NETWORK_INTERFACE].bytes_sent + netio2[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)
			net_usage = net_usage2 - net_usage1

			bandwidth = net_usage / 5
			print(str(bandwidth)+" MB/s")

			band_data = []
            band_data.append(datetime.now())
            band_data.append(each_stag)
            band_data.append(each_lps)
            band_data.append(net_usage)
            band_data.append(bandwidth)
            # common_data.append(gs)
            # common_data.append(loss_value)
            # common_data.append(duration)
            # common_data.append(accuracy)
            # common_data.append(val_accuracy)

            band_data_csv = []
            band_data_csv.append(band_data)
            import csv
            file_common = 'resnet_my/band_graph.csv' 
            with open(file_common, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(band_data_csv)


	# command = "ping gpusrv05.cs.virginia.edu"
	# p = Process(target=get_bandwidth, args=(command,))
	# p.start()
	# NETWORK_INTERFACE = 'enp1s0f0'
	# netio1 = psutil.net_io_counters(pernic=True)
	# net_usage1 = (netio1[NETWORK_INTERFACE].bytes_sent + netio1[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)

	# time.sleep(5)
	# # p.join()
	# netio2 = psutil.net_io_counters(pernic=True)
	# net_usage2 = (netio2[NETWORK_INTERFACE].bytes_sent + netio2[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)
	# net_usage = net_usage2 - net_usage1

	# bandwidth = net_usage / 5
	# print(str(bandwidth)+" MB/s")



if __name__ == '__main__':
	main()



