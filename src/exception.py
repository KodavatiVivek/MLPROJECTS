import sys

def error_message_details(error):
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:  # Check if there's an actual traceback
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    return f"Error occurred in script: {file_name} at line number: {line_number}, error message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message)

    def __str__(self):
        return self.error_message
 
