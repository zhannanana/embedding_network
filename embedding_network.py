import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import torch.optim as optim
import pandas as pd


class ETN(object):
    def __init__(self, user_seq):
        self.user_seq = user_seq
        self.u_id_list = list(self.user_seq.keys())
        # print("len:", len(self.u_id_list))
        self.user_embedding = [torch.randn(1, 200, requires_grad=True) for item in range(10000)]

        # print("user_embedding[24]", self.user_embedding[24])
        # print("user_embedding", self.user_embedding)
        # self.user_embedding = nn.Embedding(len(self.u_id_list), 200)  # 200维
        # self.e_index = torch.LongTensor([item for item in range(len(self.u_id_list))])
        # self.e_idx = Variable(self.e_index)  #二维向量tensor
        # self.u_emd = [self.user_embedding(iidx) for iidx in self.e_idx]

    def distance(self, x_id, y_id):
        # print("in x_id:", x_id, "y_id:", y_id)
        # e1 = (self.user_embedding[int(x_id)]).detach().numpy()   # tensor、Variable
        # e2 = (self.user_embedding[int(y_id)]).detach().numpy()
        # self.user_embedding[int(x_id)]
        # return -np.sum(np.square(e1 - e2))
        #return torch.dist(self.user_embedding[int(x_id)], self.user_embedding[int(y_id)])
        res = torch.sqrt(torch.sum((self.user_embedding[int(x_id)]-self.user_embedding[int(y_id)]) ** 2))
        # print("distance:", res)
        return res

    def function_k(self, time, time_h):
        # return math.exp(-delta_s(time-time_h))  # 值可以调整
        res = math.exp(0.0000001*(time - time_h))
        # print("res:", res)
        return res


    # 公式3 和 5(5应该也不要了，因为后面公式10不用)
    def lambda_xy(self, x_id, y_id, time):
        u_x_y = self.distance(x_id, y_id)  # x_id y_id对应的embedding的欧氏距离
        # T时刻以前所有的影响,按照固定长度的序列
        before_sum = 0
        u_list = self.user_seq[x_id]

        # 计算wight的分母
        wight_denominator = 0
        for i in range(len(u_list)):
            h_id = u_list[i][1]
            wight_denominator += torch.exp(self.distance(h_id, x_id))
            if wight_denominator == 0:
                print("wight_denominator == 0 error")

        for i in range(len(u_list)):
            h_time = u_list[i][0]
            h_id = u_list[i][1]
            wight = torch.exp(self.distance(h_id, x_id)) / wight_denominator
            alpha_h_y = wight * self.distance(h_id, y_id)

            k = self.function_k(time, h_time)
            before_sum += alpha_h_y * k
        # print("uxy:", u_x_y)
        # print("before_sum:", before_sum)
        # print("uxy + before_sum:", u_x_y+before_sum)
        print('lambda:', u_x_y + before_sum)
        return u_x_y + before_sum



    # 公式10
    def function_log(self, x_id, y_id, time, k):  # k个负样本
        l_xy = self.lambda_xy(x_id, y_id, time)
        log_q = torch.log((1/(1+torch.exp(-l_xy))))
        print("log_q:", log_q)
        length = len(self.u_id_list)
        #print("length :", length)
        x_list =[]
        user_list = self.user_seq[x_id]
        for i in range(len(user_list)):
            x_list.append(user_list[i][1])  # 把用户id,加进来

        sample_list = []
        for i in range(k):
            count = 0
            while count < length:    # 这里可能还要根据大小修改
                count += 1
                sample_id = random.randint(0, length-1)
                #print("sample id:", sample_id)
                if self.u_id_list[sample_id] in x_list:
                    continue
                else:
                    sample_list.append(sample_id)
                    break

        if len(sample_list) < k:
            print("没有找到足够的负样本")

        sum_log = 0
        for i in range(len(sample_list)):
            t_y_id = sample_list[i]
            l_temp = self.lambda_xy(x_id, t_y_id, time)
            t_log = torch.log((1/(1+torch.exp(-l_temp))))
            sum_log += t_log
        # print("log_q:", log_q)
        # print("sum_log:", sum_log)
        return log_q + sum_log


def get_user_seq():
    # 定义时间间隔为2小时以内的算是同一个时段60*60*2 = 7200秒

    csv_file = "allchickinscsv.csv"
    csv_data = pd.read_csv(csv_file, low_memory=False)
    csv_df = csv_data.head(10000)  # 这样就不用上面哪些json了

    user_seq = {}  # 先定义成dict然后可以再转成df
    uid_list = csv_df['uid']
    uid_set = set(uid_list)
    for uid in uid_set:
        # 遍历用户uid的签到记录
        one_df = csv_df[csv_df.uid == uid]
        temp_list = []
        for row in one_df.iterrows():
            pid = row[1]['pid']
            time = row[1]['time']

            temp_df = csv_df[csv_df.pid == pid]
            sign_df = temp_df[abs(temp_df.time-time) <= 7200]

            res_uid = sign_df['uid']
            for one_uid in res_uid:
                if one_uid == uid:
                    continue
                item = [time, one_uid]
                temp_list.append(item)
        temp_dict = {uid: temp_list}
        user_seq.update(temp_dict)
    return user_seq


user_seq = get_user_seq()
etn = ETN(user_seq)
echopes = 10
# 这里为什么不变呢
optimizer = optim.SGD(etn.user_embedding, lr=0.0001, momentum=0.9)
#print("etn.user_embedding：", etn.user_embedding)

for echop in range(echopes):
    optimizer.zero_grad()

    for item in user_seq.keys():
        x_id = item
        if user_seq[x_id] == []:
            continue
        y_id = user_seq[x_id][-1][1]
        time = user_seq[x_id][-1][0]
        print("x:", x_id, "y:", y_id)
        loss = etn.function_log(x_id, y_id, time, 5)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("user:", item, "loss:", loss)
#print("etn.user_embedding：", etn.user_embedding)
















