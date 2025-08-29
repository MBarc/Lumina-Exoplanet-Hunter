import time
import threading
import logging
from pymongo import MongoClient
import json
import os
import socket

def on_startup():
    logging.info("Data Gathering Service is starting up...")
    # Perform any initialization tasks here

    # check config file

    # check log directory

    # check db connection

    # Check db["sectors"]["processing"] to see if there are any sectors that are in the processing stage that have this machine's hostname as the worker
        # if yes, continue working on that sector.
            # for each TIC ID
                # if TIC ID status is "downloaded", move onto next
                # else continue working on this TIC ID, then move onto next
        # if no, check the queue for new sector
    return

def check_db_connection(config, dbConn):
    try:
        dbConn.admin.command("ping")
        logging.info(f"Connected to MongoDB on {config['db_host']}:{config['db_port']} as {config['db_username']}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return False

def check_config(config_path=r"C:\\Program Files\\ExoplanetHunter\\Data\\config\\config.json"):
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from config file: {config_path}")
        return None

    return config

def check_log_directory(log_dir):
    if not os.path.exists(log_dir):
        logging.error(f"Log directory does not exist: {log_dir}")
        return False
    return True

def is_sector_queue_empty(dbConn):
    return dbConn["sectors"]["queue"].count_documents({}) == 0

def run_data_gathering(stop_event, config_path=r"C:\\Program Files\\ExoplanetHunter\\Data\\config\\config.json"):
    """
    Main entrypoint for data gathering, runs until stop_event is set.
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    log_dir = config.get("log_directory", ".")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "data_gathering_logs.txt")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Data Gathering process started")

    try:
        dbConn = MongoClient(
            host=config["db_host"],
            port=int(config["db_port"]),
            username=config["db_username"],
            password=config["db_password"],
            authSource=config["db_auth_database"],
            serverSelectionTimeoutMS=5000  # fail fast (5 seconds)
        )
        # Force connection
        dbConn.admin.command("ping")
        logging.info(f"Connected to MongoDB on {config['db_host']}:{config['db_port']} as {config['db_username']}")

        if not is_sector_queue_empty(dbConn):
            logging.info("Sector queue is not empty, proceeding with data gathering...")

            # Getting the first entry in the list
            targetSector = dbConn["sectors"]["queue"].find_one({})
            logging.info(f"Identified target Sector {targetSector['sector_number']} for processing...")

            # Creating entry in ["sectors"]["processing"]
            dbConn["sectors"]["processing"].insert_one({
                "data": targetSector,
                "worker": socket.gethostname(),
                "start_time": time.time() # the current time
            })

            logging.info(f"Sector {targetSector['sector_number']} has now been moved to the processing queue.")

            dbConn["sectors"]["queue"].delete_one({"_id": targetSector["_id"]})
            logging.info(f"Sector {targetSector['sector_number']} has been removed from the queue since it is now being processed.")

        else:
            logging.info("Sector queue is empty, nothing to process.")

    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return

    # Example main loop (replace with your actual logic)
    while not stop_event.is_set():
        try:
            # Do the work here (fetch queue, download, etc.)
            logging.info("Checking queue...")
            # simulate work
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error during data gathering loop: {e}")

    logging.info("Stop event received, finishing current task and shutting down...")
    dbConn.close()