import argparse

from telegram_logging import TelegramHandler, TelegramFormatter
import sys
import sys
sys.path.append("..")
from data.dict import user_dict
import logging


def send2bot(msg, title):
    handler = TelegramHandler(bot_token=user_dict['token'],
                              chat_id=str(user_dict['chat_id']))
    formatter = TelegramFormatter(
        fmt="time: %(asctime)s\n%(name)s %(levelname)8s\nprocess status: %(message)s", datefmt="%d/%m/%Y %H:%M:%S", use_emoji=True)
    handler.setFormatter(formatter)
    bot = logging.getLogger(f'process name: {title}')
    bot.addHandler(handler)
    bot.setLevel(logging.INFO)
    bot.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--msg', type=str, default='msg')
    parser.add_argument('--title', type=str, default='title')
    args = parser.parse_args()
    send2bot(msg=args.msg, title=args.title)
    print()
