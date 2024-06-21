import argparse
import time

from telegram_logging import TelegramHandler, TelegramFormatter
import sys
import sys

sys.path.append("..")
from data.dict import user_dict
import logging

import socket


def get_host_ip():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception as e:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def send2bot(msg, title):
    handler = TelegramHandler(bot_token=user_dict['token'],
                              chat_id=str(user_dict['chat_id']))
    formatter = TelegramFormatter(
        fmt="time: %(asctime)s\n%(name)s %(levelname)8s\nprocess status: %(message)s", datefmt="%d/%m/%Y %H:%M:%S",
        use_emoji=True)
    handler.setFormatter(formatter)
    bot = logging.getLogger(f'process name: {title}')
    bot.addHandler(handler)
    bot.setLevel(logging.INFO)
    bot.info(msg)


if __name__ == "__main__":
    old_ip = get_host_ip()
    send2bot(f"current ip: {old_ip}", "IP")
    while True:
        new_ip = get_host_ip()
        if old_ip == new_ip:
            time.sleep(60)
        else:
            old_ip = new_ip
            send2bot(f"current ip: {old_ip}", "IP")
