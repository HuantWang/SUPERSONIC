import os, sys
import re


def kill_pid(pid):
    with os.popen(f"netstat -nutlp | grep  {pid}") as r:
        result = r.read()
    # print(f"The process under {pid} id are as follows:\n", result)
    PID = []
    for line in result.split("\n"):
        if r"/" in line:
            PID.extend(re.findall(r".*?(\d+)\/", line))  # 找到进程对应的PID

    PID = list(set(PID))
    # print(f"找到的PID={PID}")
    for pid in PID:
        try:
            os.system(f"kill {pid}")
            print(f"{pid} have been cleaned...")
        except Exception as e:
            print(e)


kill_pid(50055)
