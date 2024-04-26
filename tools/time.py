import time
from datetime import datetime
import requests
import pytz


def get_hour():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.hour


def get_minute():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.minute


if __name__ == '__main__':
    print(get_minute())
