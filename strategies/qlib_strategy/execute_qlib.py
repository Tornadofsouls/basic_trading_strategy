#coding:gbk
import requests
import pandas
import time
import math
import datetime
import json
from io import BytesIO

class StrategyState():
    trade_complete = [False]
    trade_message = []
    target_holding_stocks = {}

    def add_trade_message(self, message):
        self.trade_message.append(message)
        print(message)

def get_pre_trading_date():
    calendar_csv_url = "https://trade-1254083249.cos.ap-nanjing.myqcloud.com/calendar/sse.csv"
    response = requests.get(calendar_csv_url)
    if response.status_code != 200:
        raise Exception("failed to get score csv", calendar_csv_url, " status_code ", response.status_code)
    calendar_list = pandas.read_table(BytesIO(response.content),sep=",").astype({"cal_date": "str", "pretrade_date": "str"})
    today_entry = calendar_list[calendar_list["cal_date"] == datetime.datetime.today().strftime("%Y%m%d")]
    if len(today_entry) == 0:
        return None
    pretrade_date_str = today_entry.iloc[0]["pretrade_date"]
    pretrade_date_obj = datetime.datetime.strptime(pretrade_date_str, "%Y%m%d")
    return pretrade_date_obj

def get_last_day_pred_table():
    last_trade_date = get_pre_trading_date()
    if last_trade_date is None:
        return None
    score_csv_url = "https://trade-1254083249.cos.ap-nanjing.myqcloud.com/predict/{}.csv".format(last_trade_date.strftime("%Y-%m-%d"))
    response = requests.get(score_csv_url)
    if response.status_code != 200:
        raise Exception("failed to get score csv", score_csv_url, " status_code ", response.status_code)
    score_list = pandas.read_table(BytesIO(response.content),sep=",")
    score_list["stockcode"] = score_list["instrument"].str[2:8] + "." + score_list["instrument"].str[:2]
    return score_list

def init(ContextInfo):
    ContextInfo.accID = '102666'
    ContextInfo.last_day_pred = get_last_day_pred_table()
    ContextInfo.sell_white_list = [] # Do not sell these stocks
    ContextInfo.state = StrategyState()

def send_notification_log(title, text):
    print("Logging to azure", title, text)
    url = "https://di-trading-log.azurewebsites.net/api/log_event?code=gMcbj7J1vKh/VCs8e2MkVaHRp/4NLz8dttpgk03p8SMKcJQHo/8JKQ=="
    data = {
        "title" : title,
        "desp" : text,
    }
    try_cnt = 0
    while try_cnt < 5:
        try:
            response = requests.post(url, data=json.dumps(data))
            print(response.content)
            return
        except Exception as e:
            print(e)
            try_cnt += 1


last_price_cache = {}

def get_last_price(ContextInfo, stock_code):
    global last_price_cache 
    if stock_code not in last_price_cache:
        last_price = ContextInfo.get_market_data(["close"], stock_code = [stock_code], skip_paused = True, period = '1m', dividend_type = 'none', count = -1)
        last_price_cache[stock_code] = last_price
        print(stock_code, "Last price is", last_price_cache[stock_code])
    return last_price_cache[stock_code]
def handlebar(ContextInfo):
    if ContextInfo.state.trade_complete[0]:
        return

    if not ContextInfo.is_last_bar():
        return 

    if not ContextInfo.is_new_bar():
        return

    if ContextInfo.last_day_pred is None:
        send_notification_log("Skip non trading day", "QLIB_TRADE_COMPLETE")
        return 

    in_trade_time = True
    current_time_str = datetime.datetime.now().strftime("%H%M")
    if current_time_str < "0930" or current_time_str > "1500":
        in_trade_time = False

    if current_time_str > "1130" and current_time_str < "1300":
        in_trade_time = False

    acct_info_list = get_trade_detail_data(ContextInfo.accID, 'stock', 'account')
    if len(acct_info_list) != 1:
        raise Exception(f"Incorrect account number {len(acct_info_list)}")
    acct_info = acct_info_list[0]
    print("Total balance: ", acct_info.m_dBalance)
    print("Available balance: ", acct_info.m_dAvailable)
    balance_left = acct_info.m_dAvailable

    total_trategy_balance = acct_info.m_dBalance - 10000
    #if total_trategy_balance > acct_info.m_dBalance - 10000:
    #    total_trategy_balance = acct_info.m_dBalance - 10000
    print("Strategy balance:", total_trategy_balance)

    # Get current holding stocks
    curr_holding = {}
    position_list = get_trade_detail_data(ContextInfo.accID, "STOCK", "POSITION")
    two_month_ago = (datetime.datetime.today() - datetime.timedelta(days=60)).strftime("%Y%m%d")
    for position in position_list:
        stock_code = f"{position.m_strInstrumentID}.{position.m_strExchangeID}"
        open_date = ContextInfo.get_open_date(stock_code)
        print(open_date, two_month_ago, stock_code)
        if position.m_nVolume == 0:
            continue
        if str(open_date) == "19700101" or str(open) > two_month_ago:
            # Skip new stock
            continue
        curr_holding[stock_code] = {
            "name" : position.m_strInstrumentName,
            "vol": position.m_nVolume,
        }

    # Get target holding stocks and corresponding volume
    score_list = ContextInfo.last_day_pred
    target_holding_stocks = ContextInfo.state.target_holding_stocks
    if len(target_holding_stocks) == 0:
        strategy_residual_balance = 0
        for index, stock_info in score_list.iterrows():
            stock_code = stock_info["stockcode"]
            min_shares = 100
            if stock_code.startswith("688"):
                min_shares = 200
            # Add to target holding list
            last_price = get_last_price(ContextInfo, stock_code)
            # Do not use existing volume, as volume might not reflect order status
            target_balance = total_trategy_balance / 10 
            target_volume = math.floor(target_balance / last_price / min_shares) * min_shares
            
            strategy_residual_balance += (total_trategy_balance / 10) - (target_volume * last_price)
                
            target_holding_stocks[stock_code] = target_volume
            if len(target_holding_stocks) == 10:
                break

        print("Assigning residual balance:", strategy_residual_balance)
        # Attribute residual balance
        while strategy_residual_balance > 0:
            # Calculate diff with average target
            # Attribute to largest diff
            target_single_stock_amount = total_trategy_balance / 10 
            amount_diff_map = {}
            for stock_code, stock_volume in target_holding_stocks.items():
                last_price = get_last_price(ContextInfo, stock_code)
                amount_diff_map[stock_code] = target_single_stock_amount - (last_price * stock_volume)
            print(amount_diff_map)

            largest_diff_stock = max(amount_diff_map, key=amount_diff_map.get)
            min_shares = 100
            if largest_diff_stock.startswith("688"):
                min_shares = 200

            min_amount = get_last_price(ContextInfo, largest_diff_stock) * min_shares
            # Insufficient average target
            if min_amount > strategy_residual_balance:
                break

            strategy_residual_balance -= min_amount
            target_holding_stocks[largest_diff_stock] += min_shares

        print("Residual balance:", strategy_residual_balance)
    trade_complete = True

    # outstanding orders
    order_list = get_trade_detail_data(ContextInfo.accID, "STOCK", "ORDER")
    stock_code_already_order = {}
    for order in order_list:
        # Skip cancel order, ENTRUST_STATUS_CANCELED
        # Skip complete order
        if order.m_nOrderStatus in [51, 52, 53, 54, 56, 57]:
            continue
        stock_code = "{}.{}".format(order.m_strInstrumentID, order.m_strExchangeID)
        stock_code_already_order[stock_code] = order

    # Compare and generate orders 
    # Sell stock which is not in target list
    for stock_code in curr_holding.keys():
        if stock_code in ContextInfo.sell_white_list:
            continue
        if stock_code in stock_code_already_order:
            # Wait for order to ENTRUST_STATUS_SUCCEEDED
            print("Order for ", stock_code, " not completed, canceling")
            trade_complete = False
            cancel(str(stock_code_already_order[stock_code].m_nRef),ContextInfo.accID,'STOCK',ContextInfo)
        if curr_holding[stock_code]["vol"] == 0:
            continue
        if stock_code not in target_holding_stocks:
       #     up_stop_price = ContextInfo.get_instrumentdetail(stock_code)["UpStopPrice"],
       #     down_stop_price =  ContextInfo.get_instrumentdetail(stock_code)["DownStopPrice"],
            trade_complete = False
            stock_name = ContextInfo.get_instrumentdetail(stock_code)["InstrumentName"]
            sell_volume = curr_holding[stock_code]["vol"]
            if not in_trade_time:
                down_stop_price =  ContextInfo.get_instrumentdetail(stock_code)["DownStopPrice"]
                passorder(24,1101,ContextInfo.accID,stock_code,11,down_stop_price,sell_volume,"qlib",1,ContextInfo)
            else:
                passorder(24,1101,ContextInfo.accID,stock_code,4,-1,sell_volume,"qlib",1,ContextInfo)
            trade_msg = "Sell {} {} shares: {}".format(stock_code, stock_name, sell_volume)
            ContextInfo.state.add_trade_message(trade_msg)
    # Buy stock in target list
    for stock_code, target_volume in target_holding_stocks.items():
        if stock_code in stock_code_already_order:
            print("Order for ", stock_code, " not completed, canceling")
            trade_complete = False
            cancel(str(stock_code_already_order[stock_code].m_nRef),ContextInfo.accID,'STOCK',ContextInfo)
        curr_vol = curr_holding.get(stock_code, {"vol": 0})["vol"]
        if target_volume != curr_vol:
            trade_complete = False
            buy_volume = target_volume - curr_vol
            stock_name = ContextInfo.get_instrumentdetail(stock_code)["InstrumentName"]
            last_price = get_last_price(ContextInfo, stock_code)
            if (buy_volume * last_price) > balance_left:
                print("No enough balance for", stock_code, "balance left", balance_left, "need", buy_volume * last_price)
                continue
            if buy_volume * last_price > 0:
                balance_left -= buy_volume * last_price
            if in_trade_time:
                order_type = 23 if buy_volume > 0 else 24
                price_type =  6 if buy_volume > 0 else 4
                order_vol = buy_volume if buy_volume > 0 else 0 - buy_volume
                #passorder(order_type,1101,ContextInfo.accID,stock_code,5,-1,buy_volume,"qlib", 1,ContextInfo)
                #smart_algo_passorder(order_type,1101,ContextInfo.accID,stock_code,5,-1,buy_volume,"qlib", 1,"qlib","FLOAT",0,0,ContextInfo)
                #orderParams = {"OrderType": 1, "PriceType": 6}
                #algo_passorder(order_type,1101,ContextInfo.accID,stock_code,6,-1,buy_volume,"qlib", 1,"qlibid", orderParams,ContextInfo)
                passorder(order_type,1101,ContextInfo.accID,stock_code,price_type,-1,order_vol,"qlib",1,ContextInfo)
            trade_msg = "Buy {} {}  shares: {}".format(stock_code, stock_name, target_volume - curr_vol)
            ContextInfo.state.add_trade_message(trade_msg)

    if trade_complete:
        ContextInfo.state.trade_complete[0] = True
        title = "All trade completed"
        ContextInfo.state.add_trade_message("QLIB_TRADE_COMPLETE")
        print("\n".join(ContextInfo.state.trade_message))
        send_notification_log(title, "\n".join(ContextInfo.state.trade_message))
