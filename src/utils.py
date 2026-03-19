import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        """
        :param error_message: The custom error message string
        :param error_details: The 'sys' module to extract traceback info
        """
        super().__init__(error_message) # Always call the base class init
        
        _, _, exc_tb = error_details.exc_info()
        
        # Safety check: ensure traceback exists before accessing attributes
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "Unknown"
            self.file_name = "Unknown"
            
        self.error_message = error_message

    def __str__(self):
        return f"Error occurred in: [{self.file_name}] line: [{self.lineno}] message: [{self.error_message}]"
