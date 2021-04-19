# -*- coding: utf-8 -*-

import requests
import glob
import os


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def main(ip, port, sid, token, ans, problem):
    print("正在提交...")
    url = "http://%s:%s/jsonrpc" % (ip, port)

    payload = {
        "method": problem,
        "params": [ans],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()

    print(response)
    if "auth_error" in response:
        print("您的认证信息有误")
        return response["auth_error"]
    elif "error" not in response:
        print("测试完成，请查看分数")
        return response["result"]
    else:
        print("提交文件存在问题，请查看error信息")
        return response["error"]["data"]["message"]


if __name__ == "__main__":
    # 需要修改的参数：problem, name

    # problem 参数：
    #    VI_evaluate: 汽车保险预测
    #    SP_evaluate: 学生成绩预测
    #    SUMM_evaluate: 问题摘要生成
    #    DK_evaluate: 贷款资格审查
    #    QY_evaluate: 球员能力评测
    #    DZ_evaluate: 视频动作识别
    #    DM_evaluate: 动漫Face检测
    problem = "DZ_evaluate"
    # IP 不需要修改
    ip = "81.70.0.150"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = "BY2006192"
    # 改成你的口令
    token = "666"
    path ='Action/submission.txt'
    if problem in ["VI_evaluate", "SP_evaluate", "SUMM_evaluate"]:
        import json
        # 你的JSON文件
        with open(path) as f:
            d = json.load(f)
    elif problem in ["DK_evaluate", "QY_evaluate"]:
        import numpy as np
        d = np.loadtxt('submission.txt').tolist()
    elif problem == "DZ_evaluate":
        with open(path) as f:
            d = f.readlines()
    elif problem == "DM_evaluate":
        dr_files_list = glob.glob('submission' + '/*.txt')
        dr_files_list.sort()

        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line

                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append(
                    {"confidence": confidence, "file_id": file_id, "bbox": bbox}
                )

        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        d = bounding_boxes

    score = main(ip, port, sid, token, d, problem)
    print(score)