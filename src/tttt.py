import datetime
import post_1office



ma_nhan_vien = "VTCODE02402"

a  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(a, int(ma_nhan_vien[6:]))
# post_1office.post_checkin_1office())