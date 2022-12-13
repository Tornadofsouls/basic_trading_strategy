import requests
import json

def check_log_exists(start_time="9h", end_time="10h"):
  appId = "10304985-784e-4aa7-b623-1b9d8d9b47cc"
  appKey = "xxx"

  query = """
let StartTime = todatetime("2022-12-02 20:02:20");
let EndTime = todatetime("2022-12-15 21:02:20");
  traces
  | where timestamp between (StartTime .. EndTime)
  | where message contains "IC-SpreadRollingStrategyBackTestingWrapper"
  """

  params = {"query": query}
  headers = {'X-Api-Key': appKey}
  url = f'https://api.applicationinsights.io/v1/apps/{appId}/query?timespan=P7D'

  response = requests.get(url, headers=headers, params=params)

  logs = json.loads(response.text)
  columns = [x['name'] for x in logs['tables'][0]['columns']]
  rows = logs['tables'][0]['rows']
  print(columns)
  print(rows)
  for row in rows:
    structured = dict(zip(columns, row))
    print(structured)

check_log_exists()