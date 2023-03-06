import os
import numpy as np
import matplotlib.pyplot as plt
import random


def generate(case_name, instance_path='../all_data_set/', change_rate=0.2, bound=0.1):
    instance_file = instance_path + case_name + ".txt"
    with open(instance_file, 'r') as f:
        user_line = f.readline()
        data = user_line.split('\t')
        m_n = list(map(int, data))
        data = f.read()
        data = str(data).replace('\n', '\t')
        data = str(data).split('\t')
        if data.__contains__(""):
            data.remove("")
        job = list(map(int, data))
        job = np.array(job).reshape(m_n[0], m_n[1] * 2)
    f.close()
    job_num = m_n[0]
    machine_num = m_n[1]
    max_op_len = 0
    # find maximum operation length of all jobs
    for j in range(job_num):
        for i in range(machine_num):
            if max_op_len < job[j][i * 2 + 1]:
                max_op_len = job[j][i * 2 + 1]
    for j in range(job_num):
        for i in range(machine_num):
            if random.random() < change_rate:
                process_time = job[j][i * 2 + 1]
                if bound <= 1:
                    new_process_time = round(process_time*random.uniform(1-bound, 1+bound))
                else:
                    new_process_time = round(random.uniform(1, max_op_len))
                job[j][i * 2 + 1] = new_process_time
    output_path = "uniform_time_large/"
    new_instance = output_path + case_name + "_" + str(int(change_rate*100)) + "_" + str(int(100*bound)) + ".txt"
    file = open(new_instance, mode='w')
    file.write(str(job_num)+'\t')
    file.write(str(machine_num))
    file.write('\n')
    for j in range(job_num):
        jobi = []
        for i in range(machine_num):
            time = job[j][i * 2 + 1]
            machine = job[j][i * 2]
            jobi.append(str(machine))
            jobi.append('\t')
            jobi.append(str(time))
            jobi.append('\t')
        jobi.append('\n')
        file.writelines(jobi)
    file.close()


if __name__ == '__main__':
    path = 'data_set_standard/'
    for file_name in os.listdir(path):
        print(file_name + "========================")
        title = file_name.split('.')[0]
        for m in range(2):
            for n in range(5):
                generate(title, path, change_rate=(m+1)*0.25, bound=(n+1)*0.2)



