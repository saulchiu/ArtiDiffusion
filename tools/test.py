import time

# 获取当前时间戳
current_time = time.time()
time_tuple = time.localtime(current_time)
tag = str(time.strftime("%Y%m%d%H%M%S", time_tuple))
print(tag)