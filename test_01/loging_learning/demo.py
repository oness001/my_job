from loging_learning.log_dec import Logger

x = Logger.Logger("debug")

x.critical("这是一个 critical 级别的问题！")
x.error("这是一个 error 级别的问题！")
x.warning("这是一个 warning 级别的问题！")
x.info("这是一个 info 级别的问题！")
x.debug("这是一个 debug 级别的问题！")

x.log(50, "这是一个 critical 级别的问题的另一种写法！")
x.log(40, "这是一个 error 级别的问题的另一种写法！")
x.log(30, "这是一个 warning 级别的问题的另一种写法！")
x.log(20, "这是一个 info 级别的问题的另一种写法！")
x.log(10, "这是一个 debug 级别的问题的另一种写法！")

x.log(51, "这是一个 Level 51 级别的问题！")
x.log(11, "这是一个 Level 11 级别的问题！")
x.log(9, "这条日志等级低于 debug，不会被打印")
x.log(0, "这条日志同样不会被打印")

