import configparser
import logging
import os

config = configparser.ConfigParser()
config.read("./config.ini")
if config['Environment'].get('running_machine') == 'colab':
    log_file_path = config['Logging'].get('log_file_path_colab')
else:
    log_file_path = config['Logging'].get('log_file_path_computer')

if not os.path.isdir(os.path.dirname(log_file_path)):
    os.makedirs(os.path.dirname(log_file_path))

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(log_file_path)
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)