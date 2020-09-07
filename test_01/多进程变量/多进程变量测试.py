from multiprocessing import Queue

q = Queue(10)  # 生成一个队列对象
# put方法是往队列里面放值
q.put('Cecilia陈')
q.put('xuchen')
q.put('喜陈')

# get方法是从队列里面取值
print(q.get())
print(q.get())
print(q.get())

q.put(5)
q.put(6)
print(q.get())


from multiprocessing import Manager,Process

m = Manager().list()