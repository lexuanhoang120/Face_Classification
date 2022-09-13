from subprocess import check_output
import requests
from datetime import datetime

def post_checkin_1office(code):
    # code : ma nhan vien
    # data: ngay_checkin_checkout
#     date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(date)
    # date = "2022-08-18 16:35:55"
    url = "https://space.1office.vn/timekeep/attendance/service"
    data = '''[{ '''+ f'''
             "code": "{code}", 
            "time":"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",
            "machine_code":"VP", 
            "machine_ip":"192.168.0.201" ''' + '''}]'''
    payload= { 
            'key': 'space', 
            'data': data
            }
    files=[]
    # headers = { 'Cookie': 'PHPSESSID=6can653vsteb4s83nqar5c458n' }
    headers = {}
    requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(re.text)
    del url, data, payload, files, headers
    return 0

def post_checkout_1office(code,date):
    # code : ma nhan vien
    # data: ngay_checkin_checkout
#     date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(date)
    # date = "2022-08-18 16:35:55"
    url = "https://space.1office.vn/timekeep/attendance/service"
    data = '''[{ '''+ f'''
             "code": "{code}", 
            "time":"{date}",
            "machine_code":"VP", 
            "machine_ip":"192.168.0.201" ''' + '''}]'''
    payload= { 
            'key': 'space', 
            'data': data
            }
    files=[]
    # headers = { 'Cookie': 'PHPSESSID=6can653vsteb4s83nqar5c458n' }
    headers = {}
    re = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(re.text)
    del url, data, payload, files, headers, re
    return 0
# post_checkout_1office(2401,"2022-08-22 08:02:55")

