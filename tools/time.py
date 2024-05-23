import time
from datetime import datetime
import requests
import pytz
import logging

logger = logging.getLogger('cat')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_hour():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.hour


def get_minute():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.minute


def now():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    formatted_time = bj_time.strftime('%Y%m%d%H%M')
    return formatted_time


def sleep_cat():
    while True:
        current_hour = get_hour()
        current_minute = get_minute()
        if current_hour == 8 and current_minute >= 30:
            # when time is 8:30 or later, we should sleep the process
            current_hour += 1
        if current_hour in range(0, 9) or current_hour in range(22, 24):
            logger.info('cat is awaking')
            break
        else:
            logger.info('cat is sleeping')
            if (22 - get_hour()) > 1:
                time.sleep(3600)
            else:
                time.sleep(600)


if __name__ == '__main__':
    print(now())
