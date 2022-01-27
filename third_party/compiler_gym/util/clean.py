import os
import re

def cleanpid(portnum: str):
    #TODO:
    """ information
    :param portnum: The port number
    """
    with os.popen(f'netstat -nutlp | grep '+portnum ) as r:
        result = r.read()
    PID = []
    for line in result.split("\n"):
        if r"/" in line:
            PID.extend(re.findall(r".*?(\d+)\/", line))
    with os.popen(f'pgrep -x "stoke_search"') as r:
        result = r.read()
    for _ in result.split("\n"):
        if _ !='':
            PID.append(_)

    PID = list(set(PID))
    # print(f"PID:{PID}")
    for pid in PID:
        try:
            os.system(f"kill -9 {pid}")
        except Exception as e:
            print(e)
    print("All PID has been cleaned")

# test
# cleanpid("50001")