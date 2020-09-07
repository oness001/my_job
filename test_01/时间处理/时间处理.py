#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Datetime : 2017/11/23 下午12:37
# @Author   : Alfred Xue
# @E-Mail   : Alfred.Hsueh@gmail.com
# @GitHub   : https://github.com/Alfred-Xue
# @Blog     : http://www.cnblogs.com/alfred0311/

import datetime
import time


# 日期时间字符串
st = "2017-11-23 16:10:10"
# 当前日期时间
dt = datetime.datetime.now()
# 当前时间戳
sp = time.time()

# 1.把datetime转成字符串
def datetime_toString(dt):
    print("1.把datetime转成字符串: ", dt.strftime("%Y-%m-%d %H:%M:%S"))


# 2.把字符串转成datetime
def string_toDatetime(st):
    print("2.把字符串转成datetime: ", datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S"))


# 3.把字符串转成时间戳形式
def string_toTimestamp(st):
    print("3.把字符串转成时间戳形式:", time.mktime(time.strptime(st, "%Y-%m-%d %H:%M:%S")))


# 4.把时间戳转成字符串形式
def timestamp_toString(sp):
    print("4.把时间戳转成字符串形式: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sp)))


# 5.把datetime类型转外时间戳形式
def datetime_toTimestamp(dt):
    print("5.把datetime类型转外时间戳形式:", time.mktime(dt.timetuple()))


# 1.把datetime转成字符串
datetime_toString(dt)
# 2.把字符串转成datetime
string_toDatetime(st)
# 3.把字符串转成时间戳形式
string_toTimestamp(st)
# 4.把时间戳转成字符串形式
timestamp_toString(sp)
# 5.把datetime类型转外时间戳形式
datetime_toTimestamp(dt)