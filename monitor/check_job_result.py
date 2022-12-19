import requests
import json
import pandas
import logging
from io import BytesIO
from datetime import timedelta, datetime, tzinfo, timezone
from zoneinfo import ZoneInfo
import os

logger = logging.getLogger(__name__)

appId = "10304985-784e-4aa7-b623-1b9d8d9b47cc"
appKey = os.environ.get("APP_KEY")

def send_notification_log(title, text, logonly=False):
  print("Logging to azure", title, text)
  url = "https://di-trading-log.azurewebsites.net/api/log_event?code=gMcbj7J1vKh/VCs8e2MkVaHRp/4NLz8dttpgk03p8SMKcJQHo/8JKQ=="
  if logonly:
    url += "&logonly=1"
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

def today_is_trading_day():
  calendar_csv_url = "https://trade-1254083249.cos.ap-nanjing.myqcloud.com/calendar/sse.csv"
  response = requests.get(calendar_csv_url)
  if response.status_code != 200:
      raise Exception("failed to get trading day csv", calendar_csv_url, " status_code ", response.status_code)
  calendar_list = pandas.read_table(BytesIO(response.content),sep=",").astype({"cal_date": "str", "pretrade_date": "str"})
  today_entry = calendar_list[calendar_list["cal_date"] == datetime.today().strftime("%Y%m%d")]
  return len(today_entry) > 0

def check_log_exists(keyword, alert_text, start_time=datetime(2022, 12, 1, 20), end_time=datetime(2022, 12, 1, 21)):
  if not today_is_trading_day():
    print("Skipping non-trading day")
    return True

  local_tz = ZoneInfo("Asia/Shanghai")
  local_time = datetime.now(local_tz)
  start_time = start_time.replace(year=local_time.year, month=local_time.month, day=local_time.day, tzinfo=local_tz)
  end_time = end_time.replace(year=local_time.year, month=local_time.month, day=local_time.day, tzinfo=local_tz)
  if end_time > local_time:
    # Not pass today's check point, check for yesterday
    start_time -= timedelta(hours=24)

  start_time_str = start_time.isoformat() 
  end_time_str = local_time.isoformat()

  query = """
let StartTime = todatetime("{}");
let EndTime = todatetime("{}");
  traces
  | where timestamp between (StartTime .. EndTime)
  | where message contains "{}"
  """.format(start_time_str, end_time_str, keyword)
  #print(query)
  params = {"query": query}
  headers = {'X-Api-Key': appKey}
  url = f"https://api.applicationinsights.io/v1/apps/{appId}/query?timespan=P7D"

  response = requests.get(url, headers=headers, params=params)

  logs = json.loads(response.text)
  if "error" in logs:
    logger("Error when query table", logs)
    return False

  columns = [x['name'] for x in logs['tables'][0]['columns']]
  rows = logs['tables'][0]['rows']
  structured_rows = [dict(zip(columns, row)) for row in rows]

  if len(structured_rows) > 0:
    #print(structured_rows)
    print("PASS for ", keyword)
    return True
  else:
    send_notification_log("Alert", alert_text)
    print(keyword, alert_text)

def check_qlib_prediction(check_time):
  if not today_is_trading_day():
    print("Skipping non-trading day")
    return True

  local_tz = ZoneInfo("Asia/Shanghai")
  local_time = datetime.now(local_tz)
  check_time = check_time.replace(year=local_time.year, month=local_time.month, day=local_time.day, tzinfo=local_tz)
  if local_time < check_time:
    print("Not check time, pass")
    return True

  score_csv_url = "https://trade-1254083249.cos.ap-nanjing.myqcloud.com/predict/{}.csv".format(datetime.today().strftime("%Y-%m-%d"))
  response = requests.get(score_csv_url)
  if response.status_code != 200:
    send_notification_log("Alert", "Qlib 预测未执行")
    print(response)
  else:
    print("PASS for ", "qlib predict")

if __name__ == "__main__":
  check_qlib_prediction(check_time=datetime(2022, 12, 1, 19))
  check_log_exists(keyword="IC-SpreadRollingStrategyBackTestingWrapper", alert_text="滚IC策略未执行", start_time=datetime(2022, 12, 1, 20), end_time=datetime(2022, 12, 1, 21))
  check_log_exists(keyword="IF-SpreadRollingStrategyBackTestingWrapper", alert_text="滚IF策略未执行", start_time=datetime(2022, 12, 1, 20), end_time=datetime(2022, 12, 1, 21))
  check_log_exists(keyword="A股新股申购完成", alert_text="A股申购策略未执行",  start_time=datetime(2022, 12, 1, 9, 30), end_time=datetime(2022, 12, 1, 11))
  check_log_exists(keyword="A股新股跟踪:", alert_text="A股新股跟踪未执行", start_time=datetime(2022, 12, 1, 9, 30), end_time=datetime(2022, 12, 1, 11))
  check_log_exists(keyword="QLIB_TRADE_COMPLETE", alert_text="Qlib 策略未执行或卡住", start_time=datetime(2022, 12, 1, 9, 30), end_time=datetime(2022, 12, 1, 10))
  # TODO: Log heartbeat
  send_notification_log("TRADE_MONITOR_HEART_BEAT", "TRADE_MONITOR_HEART_BEAT", True)
