from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from binance.client import Client


def calculate_all(current_amount_USDT, current_amount_ETH, ETH_price):
    
    current_amount = current_amount_USDT + current_amount_ETH * ETH_price
    #Получаем данные
    df_current = dataset_for_current_pred()
    # Добавляем новую колонку
    df_current = make_new_col(df_current)
    # требуется для абсолютных значений запомнить текущую
    current_value = ETH_price
    # Рассчитываем сам интервал
    res_interval = bootstrap(df_current['change'].values, n_forcast=24, intervals_in_hour=12)*current_value
    # рассчитываем ratio_hedge
    start_price = current_value
    high_board = res_interval[0]
    low_board = res_interval[1]
    # доля долларов в ETH
    ratio_hedge = calc_ratio(start_price, high_board, low_board)
    
    ETH_in_USDT = current_amount*ratio_hedge
    
    ETH = ETH_in_USDT/current_value
    
    USDT = current_amount*(1-ratio_hedge)
    
    print('Интервал: ', res_interval[1], res_interval[0])
    print()
    print('Требуется вложить {} ETH, {} USDT'.format(ETH, USDT))



def GetHistoricalData(client, symbol, interval, fromDate, toDate):
    klines = client.get_historical_klines(symbol, interval, fromDate, toDate)
    df = pd.DataFrame(klines, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms')
    df['date'] = df.dateTime.dt.strftime("%d/%m/%Y")
    df['time'] = df.dateTime.dt.strftime("%H:%M:%S")
    df = df.drop(['dateTime', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    column_names = ["date", "time", "open","close", "high", "low", "volume"]
    df = df.reindex(columns=column_names)
    return df
    
def dataset_for_current_pred(symbol = 'ETHUSDT'):
  '''Функция возвращает последние доступные данные'''
  client = Client()
  current_date = datetime.now().strftime("%d-%m-%Y")
  yesterday = (datetime.now()+timedelta(days=-1)).strftime("%d/%m/%Y")
  tommorow = (datetime.now()+timedelta(days=+1)).strftime("%d/%m/%Y") 

  fromDate = str(datetime.strptime(yesterday, '%d/%m/%Y'))
  toDate = str(datetime.strptime(tommorow, '%d/%m/%Y'))
  symbol = symbol
  interval = Client.KLINE_INTERVAL_5MINUTE
  # посколько реализована подгрузка начиная со вчерашнего дня, разумно обрезать датасет 24 часами 24*60=1440
  df = GetHistoricalData(client, symbol, interval, fromDate, toDate)[-288:] 
  return df

def bootstrap(mass, n_forcast, intervals_in_hour):
    '''Функция бутстрэпит изменения за прошедшие часы (это длина массива) n_forcast*intervals_in_hour раз, 
    процедура повторяется 10000 раз, в резульатате мы перемножеаем значения и выдаем прогноз в каждой точке таким образом (10000 симуляций)'''
    mass_res = np.zeros((10000, 2)) # 2 - нижняя и верхняя границы для каждой симуляции
    mass = np.array(mass) # переводим в numpy
    n = len(mass) # длина выборки для обучения
    size_forecast = n_forcast*intervals_in_hour
    # случайно берем числа до n, размера n_forcast*intervals_in_hour
    indexes = np.random.randint(n, size=size_forecast*10000)
    change = mass[[indexes]]
    change = change.reshape((10000, size_forecast))
    # теперь рассомтрим куммулятивное перемножение- предсказанное поведение
    cum_change = np.cumprod(change, axis=1)
    # минимальное и максимальные значения внутри временного интервалла
    min_ = np.min(cum_change, axis=1)
    max_ = np.max(cum_change, axis=1)
    sorted_res_min = np.sort(min_)
    sorted_res_max = np.sort(max_)
    #######################
    high, low = sorted_res_max[9500], sorted_res_min[500]
    vol = max((high - 1), (1 - low))
    return np.array([1 + vol, 1 - vol])

    
def make_new_col(df):
    '''Потребуются относительные изменения'''
    print('Добавляю относительные изменения')
    mass = np.zeros(len(df))
    mass[0] = 1 # первую точку доопределяем 1
    values = df['open'].values.astype(float)
    for i in range(1, len(df)):
        mass[i] = values[i]/values[i-1]
        df['change'] = mass
    return df

def return_curent_eth(start_price, end_price, high_board, low_board, eth_pool, usd_pool):
    '''Функция возвращает количество эфира для хеджа, в долларах'''
    # Исходим из того, что равномерно размазаны именно доллары! И эфир в тиках в долларах
    if end_price >= start_price:
        if end_price >= high_board:
            # у нас только usdt
            res = 0
        else:
            p = (end_price-start_price)/(high_board-start_price) # доля проданного ETH (в долларах)
            res = eth_pool*(1-p)
    elif end_price < start_price:
        if end_price <= low_board:
            # долларов нет, все это эфир
            res = usd_pool+eth_pool # в результате имеем старый и новый эфир
        else:
            # рассчитаем сколько долларов стали ETH
            p = (start_price-end_price)/(start_price-low_board)# доля обменянных долларов
            res =  p*usd_pool+eth_pool
    return res

def calc_ratio(start_price, high_board, low_board):
    return (high_board-start_price)/(high_board-low_board)