import easytrader



user = easytrader.use("htzq_client")
path_ht = r'D:\海通证券委托\xiadan.exe'
user.prepare(user='3370042200', password='215900',exe_path = path_ht,comm_password='215900')
print('连接成功')
# user.balance
user.balance


# user = easytrader.use("yh_client")
# path_yh = r'D:\Program Files\weituo\银河证券\xiadan.exe'
# user.prepare(user='210400046652', password='215900',exe_path = path_yh,comm_password=None)
# print('连接成功')
# print(user.balance)
# user.position