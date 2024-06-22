import time
from datetime import datetime
import pytz


def get_hour():
    bj_tz = pytz.timezone('Asia/Shanghai')
    bj_time = datetime.now(bj_tz)
    return bj_time.hour


current_hour = get_hour()
if current_hour in range(10, 21):
    time.sleep(0.1)
