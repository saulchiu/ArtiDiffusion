import time
from datetime import datetime
import requests
import pytz


def get_hour():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.hour


if __name__ == '__main__':
    server = 'lab'
    if server == 'lab':
        while True:
            current_hour = get_hour()
            if current_hour in range(0, 10) or current_hour in range(22, 24):
                print('run')
            else:
                print("Sleeping for 10 minutes...")
                time.sleep(1)  # 10分钟等于600秒


