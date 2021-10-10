import os
import json
import math
import pickle
import datetime

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

if __name__ == '__main__':
    with open('stock_price_dict.pickle', 'rb') as fp:
        stock_price = pickle.load(fp)
    
    company_list = os.listdir('ReleasedDataset_mp3')

    r3_info, r7_info, r15_info = {}, {}, {}

    num_skip = 0

    for company_info in company_list:
        p = company_info.find('_')
        date = company_info[p + 1:]
        start_date = (int(date[0:4]), int(date[4:6]), int(date[6:8]))
        company_name = company_info[0:p]
        try:
            stock_data = stock_price[company_name]
            r3 = get_stock_volatility(stock_data, start_date, 3)
            r7 = get_stock_volatility(stock_data, start_date, 7)
            r15 = get_stock_volatility(stock_data, start_date, 15)
            with open(os.path.join('ReleasedDataset_mp3', company_info, 'Text.txt'), 'r') as fp:
                texts = fp.readlines()
            r3_info[company_info] = {'text': texts, 'volatility': r3}
            r7_info[company_info] = {'text': texts, 'volatility': r7}
            r15_info[company_info] = {'text': texts, 'volatility': r15}
        except:
            num_skip += 1

    with open('data/full/r3.json', 'w') as fp:
        json.dump(r3_info, fp, indent=4)
    with open('data/full/r7.json', 'w') as fp:
        json.dump(r7_info, fp, indent=4)
    with open('data/full/r15.json', 'w') as fp:
        json.dump(r15_info, fp, indent=4)

    print('Number of conferences: {}, Number of volatility unfound: {}'.format(len(company_list), num_skip))
    print('Size of json files: {}, {}, {}'.format(len(r3_info), len(r7_info), len(r15_info)))

    num_train = int(len(r3_info) * 0.8)
    all_keys = sorted([k for k in r3_info.keys()])
    train_keys = all_keys[:num_train]
    test_keys = all_keys[num_train:]
    train_r3_info, train_r7_info, train_r15_info = {}, {}, {}
    test_r3_info, test_r7_info, test_r15_info = {}, {}, {}
    
    for k in train_keys:
        train_r3_info[k] = r3_info[k]
        train_r7_info[k] = r7_info[k]
        train_r15_info[k] = r15_info[k]

    for k in test_keys:
        test_r3_info[k] = r3_info[k]
        test_r7_info[k] = r7_info[k]
        test_r15_info[k] = r15_info[k]
    
    print('Size of train dataset: {}, {}, {}, Size of test dataset: {}, {}, {}'.format(len(train_r3_info), len(train_r7_info), len(train_r15_info), len(test_r3_info), len(test_r7_info), len(test_r15_info)))
    with open('data/train/r3.json', 'w') as fp:
        json.dump(train_r3_info, fp, indent=4)
    with open('data/train/r7.json', 'w') as fp:
        json.dump(train_r7_info, fp, indent=4)
    with open('data/train/r15.json', 'w') as fp:
        json.dump(train_r15_info, fp, indent=4)

    with open('data/test/r3.json', 'w') as fp:
        json.dump(test_r3_info, fp, indent=4)
    with open('data/test/r7.json', 'w') as fp:
        json.dump(test_r7_info, fp, indent=4)
    with open('data/test/r15.json', 'w') as fp:
        json.dump(test_r15_info, fp, indent=4)
