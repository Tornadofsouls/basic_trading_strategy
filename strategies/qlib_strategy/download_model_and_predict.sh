#!/bin/bash
set -e
set -x
# Download model
/qlib_trading/coscli cp cos://trade/models/myalpha158_mymodel_20220917.ml /qlib_trading/trained_model
# Dump latest data or download from web
dateStr=$(date '+%Y-%m-%d')
url="https://github.com/chenditc/investment_data/releases/download/${dateStr}/qlib_bin.tar.gz"
test_exists=$(curl -l ${url})
if [[ $test_exists ]]
then
  cd /investment_data/
  bash /investment_data/dump_qlib_bin.sh
else
  wget $url -O qlib_bin.tar.gz
fi
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
# Run prediction
cd /qlib_trading/
python predict.py --model_path /qlib_trading/trained_model
for file in $(ls *.csv) 
do   
  /qlib_trading/coscli cp $file cos://trade/predict/$file
done
# Run backtest
yesterdayStr=$(date '+%Y-%m-%d' -d "yesterday")
echo "Yesterday is" $yesterdayStr
python backtest_recent.py --model_path /qlib_trading/trained_model --start_date "2022-11-14" --end_date $yesterdayStr
