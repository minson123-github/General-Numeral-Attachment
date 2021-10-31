import os
import json
import math
import pickle
import datetime
from tqdm import tqdm

def get_stock_volatility(stock_data, start_date, r):
    y, m, d = start_date
    date = datetime.date(y, m, d)
    trade_dates = [d.date() for d in stock_data.index]
    while date not in trade_dates:
        date = date + datetime.timedelta(days=-1)

    return_prices = []
    for k in range(1, r + 1):
        pre_date = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
        date = date + datetime.timedelta(days=1)
        while date not in trade_dates:
            date = date + datetime.timedelta(days=1)

        cur_date = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
        pre_close_price = stock_data.loc[pre_date]['Close']
        cur_close_price = stock_data.loc[cur_date]['Close']
        return_price = (cur_close_price - pre_close_price) / pre_close_price
        return_prices.append(return_price)

    avg_return_price = sum(return_prices) / r
    volatility = 0
    for return_price in return_prices:
        volatility += (return_price - avg_return_price) ** 2
    volatility = math.log(math.sqrt(volatility / r))
    return volatility

def get_price_increment(stock_data, start_date, r):
    y, m, d = start_date
    date = datetime.date(y, m, d)
    trade_dates = [d.date() for d in stock_data.index]
    while date not in trade_dates:
        date = date + datetime.timedelta(days=-1)

    begin_date = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
    begin_price = stock_data.loc[begin_date]['Close']
    end_price = 0
    for k in range(1, r + 1):
        date = date + datetime.timedelta(days=1)
        while date not in trade_dates:
            date = date + datetime.timedelta(days=1)

        cur_date = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
        cur_price = stock_data.loc[cur_date]['Close']
        end_price = cur_price

    if end_price > begin_price:
        return 1
    else:
        return 0

if __name__ == '__main__':
    with open('stock_price_dict.pickle', 'rb') as fp:
        stock_price = pickle.load(fp)
    
    company_list = os.listdir('ReleasedDataset_mp3')

    info = {}

    num_skip = 0

    for company_info in tqdm(company_list):
        p = company_info.find('_')
        date = company_info[p + 1:]
        start_date = (int(date[0:4]), int(date[4:6]), int(date[6:8]))
        company_name = company_info[0:p]
        try:
            stock_data = stock_price[company_name]
            vs = []
            for r in range(2, 31):
                vs.append(get_stock_volatility(stock_data, start_date, r))
            
            ps = []
            for r in range(2, 31):
                ps.append(get_price_increment(stock_data, start_date, r))

            with open(os.path.join('ReleasedDataset_mp3', company_info, 'Text.txt'), 'r') as fp:
                texts = fp.readlines()
            info[company_info] = {'text': texts, 'volatility': vs, 'price': ps}
        except:
            num_skip += 1

    with open('list_data/full/data.json', 'w') as fp:
        json.dump(info, fp, indent=4)

    print('Number of conferences: {}, Number of volatility unfound: {}'.format(len(company_list), num_skip))
    print('Size of json files: {}'.format(len(info)))

    num_train = int(len(info) * 0.8)
    all_keys = sorted([k for k in info.keys()])
    train_keys = all_keys[:num_train]
    test_keys = all_keys[num_train:]
    train_info = {}
    test_info = {}
    
    for k in train_keys:
        train_info[k] = info[k]

    for k in test_keys:
        test_info[k] = info[k]
    
    print('Size of train dataset: {}, Size of test dataset: {}'.format(len(train_info), len(test_info)))
    with open('list_data/train/data.json', 'w') as fp:
        json.dump(train_info, fp, indent=4)

    with open('list_data/test/data.json', 'w') as fp:
        json.dump(test_info, fp, indent=4)
