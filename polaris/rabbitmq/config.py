import os

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USERNAME = os.getenv('RABBITMQ_USERNAME', None)
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', None)
RABBITMQ_VIRTUAL_HOST = os.getenv('RABBITMQ_VIRTUAL_HOST', '/')
