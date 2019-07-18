import csv
import pandas as pd
import json


def statistics():
    poi_set = set()
    poi_conut_dict = {}
    with open("allchickinscsv.csv") as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            poi_set.add(row[1])
            if row[1] not in poi_conut_dict.keys():
                poi_conut_dict[row[1]] = 1
            else:
                poi_conut_dict[row[1]] += 1

    # 遍历dict,保存地点出现的次数，然后排序，查看前1000项吧
    count_list = []
    for item in poi_conut_dict.keys():
        count_list.append(poi_conut_dict[item])

    count_list.sort(reverse=True)
    for i in range(len(count_list)):
        if count_list[i] == 200:
            print(count_list[i])
            print(i)
        if count_list[i] == 300:
            print(count_list[i])
            print(i)
        if count_list[i] == 500:
            print(count_list[i])
            print(i)

    print("地点统计结果：")
    print(len(poi_conut_dict))
    print(len(poi_set))


def get_user_seq():
    # 定义时间间隔为2小时以内的算是同一个时段60*60*2 = 7200秒
    # poi_dict的json 键是POI的id，值是[time,uid]哪个用户在什么时间访问了
    # user_dict的json 键是userid 值是[time,pid] 在哪个时间访问了那个poi
    # user_seq 的json 键是user_id 值是[time,uid] 在哪个时间，与该用户访问了同一个地方的用户，时间考虑
    # 两小时范围内，或者更宽，更窄，time按照原有uid的时间
    # 感觉dateframe更好用啊，直接将csv转为dataframe
    csv_file = "allchickinscsv.csv"
    csv_data = pd.read_csv(csv_file, low_memory=False)
    csv_df = pd.DataFrame(csv_data)  # 这样就不用上面哪些json了

    user_seq = {}  # 先定义成dict然后可以再转成df
    uid_list = csv_df['uid']  # 有哪些用户，然后对这些用户遍历,用户编号就是从0-196590
    uid_set = set(uid_list)
    for uid in uid_set:
        # 遍历用户uid的签到记录
        one_df = csv_df[csv_df.uid == uid]
        temp_list = []
        for row in one_df.iterrows():
            pid = row[1]['pid']
            time = row[1]['time']   # 感觉这里检索很麻烦
            # 找到所有访问过这个地点的访问记录
            temp_df = csv_df[csv_df.pid == pid]
            sign_df = temp_df[abs(temp_df.time-time) <= 7200]
            # sign_df至少会找到一个他自己，但是这个要去除掉
            # 按时间顺序把sign_df中的内容加入到list中？
            res_uid = sign_df['uid']
            for one_uid in res_uid:
                if one_uid == uid:
                    continue
                item = [time, one_uid]
                temp_list.append(item)
        temp_dict = {uid: temp_list}
        user_seq.update(temp_dict)
    # 如何检查写的对不对
    print(user_seq)
    filename = 'testres.json'
    with open('1.json', 'w') as f:
        json.dump(str(user_seq), f)


get_user_seq()










