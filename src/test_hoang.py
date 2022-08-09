import sqlite3


def has_existed(ma_nhan_vien):
    path = "database//check_in.sql"
    database = sqlite3.connect(path)
    query = f"SELECT * from checkin WHERE date(checkin.datetime) == date('now','localtime') and checkin.ma_nhan_vien == '{ma_nhan_vien}'"
    a = database.execute(query)
    database.commit()
    if list(a)==[]:
        return 0
    else:
        return 1
    # print(list(a)==[])

a = has_existed("TTS005")
print(a)