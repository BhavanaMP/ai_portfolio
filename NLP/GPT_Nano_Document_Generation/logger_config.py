import logging
import logging.config


def setup_logging():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            },
        },
        'handlers': {
            # 'console': {
            #     'level': 'DEBUG',
            #     'class': 'logging.StreamHandler',
            #     'formatter': 'standard',
            # }, # Enable if you want logging statements to appear in terminal.
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': 'gpt_nano.log',
                'formatter': 'standard',
                'mode': 'w',  # Set mode to 'w' for write (start fresh each time)
            },
        },
        'loggers': {
            '': {
                'handlers': ["file"],  # ['console', 'file']  # # Enable if you want logging statements to appear in terminal.
                'level': 'DEBUG',
                'propagate': True,
            },
        }
    })

# # Call this function at the start of your main script to set up logging
# setup_logging()
