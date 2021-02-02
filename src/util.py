from sqlalchemy import create_engine
import pandas as pd


def calcGain(buy, sell):
    gain = 0
    if buy >= 0:
        if sell >= 0:
            gain = buy - sell
        else:
            gain = -(buy + abs(sell))
    else:
        if sell >= 0:
            gain = abs(buy) + sell
        else:
            if buy >= sell:
                gain = -(abs(sell) - abs(buy))
            else:
                gain = abs(buy) - abs(sell)
    return gain


def getData(pair, limit=0):
    engine = create_engine('mysql+pymysql://test:pass@aws:3307/Scroller_db')
    conn = engine.connect()
    query = ''
    if limit:
        query = "select * from ticker where marketName='USDT-BTC' order by date limit " + \
            str(limit)
    else:
        query = "select * from ticker where marketName='USDT-BTC' order by date"
    data = pd.read_sql(query, conn)

    return [{"name": pair, "data": data}]
