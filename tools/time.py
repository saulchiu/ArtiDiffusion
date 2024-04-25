import time
from datetime import datetime


def get_hour():
    now = datetime.now()
    current_hour = now.strftime("%H")
    return current_hour


if __name__ == '__main__':
    server = 'lv'
    if server == 'lab':
        while True:
            current_hour = get_hour()
            if current_hour in range(0, 10) or current_hour in range(22, 24):
                print('run')
            else:
                print("Sleeping for 10 minutes...")
                time.sleep(1)  # 10分钟等于600秒


