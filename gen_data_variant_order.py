import os
import numpy as np
import matplotlib.pyplot as plt
import random


def generate(case_name, instance_path='../all_data_set/', exchange_rate=0.2):
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

    exchange_cnt = int(job_num*machine_num*exchange_rate)
    for j in range(exchange_cnt):
        job_id = random.randint(0, job_num-1)
        op1, op2 = random.sample(range(0, machine_num-1), 2)
        if op1 >= machine_num-1 or op2 >= machine_num-1:
            print("hello")
        process_time1 = job[job_id][op1 * 2 + 1]
        machine1 = job[job_id][op1 * 2]
        job[job_id][op1 * 2 + 1] = job[job_id][op2 * 2 + 1]
        job[job_id][op1 * 2] = job[job_id][op2 * 2]
        job[job_id][op2 * 2 + 1] = process_time1
        job[job_id][op2 * 2] = machine1
    new_instance = "data_set_gen_order/" + case_name + "_" + str(int(exchange_rate*100)) + ".txt"
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
        for m in range(4):
            generate(title, path, exchange_rate=(m+1)*0.2)



