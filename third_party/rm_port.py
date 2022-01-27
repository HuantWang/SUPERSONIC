import os, sys
import re

def kill_pid(pid):
    with os.popen(f'netstat -nutlp | grep  {pid}') as r:
        result = r.read()
    print(f"使用端口号{pid}的进程如下:\n", result)

    print("传我命令...开始行刑...")
    PID = []
    for line in result.split("\n"):
        if r"/" in line:
            PID.extend(re.findall(r".*?(\d+)\/", line)) # 找到进程对应的PID

    PID = list(set(PID))
    print(f"找到的PID={PID}")
    for pid in PID:
        try:
            os.system(f"kill {pid}")
            print(f"{pid} 已被处决...")
        except Exception as e:
            print(e)
        
kill_pid(50055)
