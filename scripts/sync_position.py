#!/usr/bin/env python3
import utils.tencent_db_init

from datetime import date
import json
import logging

from opencensus.ext.azure.log_exporter import AzureLogHandler
from vnpy.trader.database import get_database

from asset_management.models import PositionHistory, CurrentPosition, TargetPosition, StrategyRunStatus
from utils.system_configs import azure_log_key
from asset_management import asset_net_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(
    connection_string=azure_log_key)
)

def init_database():
    db = get_database().db
    db.create_tables([PositionHistory, CurrentPosition, TargetPosition, StrategyRunStatus])    

 def commit_database():
    db = get_database().db
    db.execute_sql('select dolt_add("-A")')
    db.execute_sql('select dolt_commit("-m", "daily update")')
    db.execute_sql('select dolt_push("main")')

    
if __name__ == "__main__":
    init_database()
    asset_net_value.calculate_latest_position_history()
    commit_database()
    
