import os

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USERNAME = os.getenv('RABBITMQ_USERNAME', None)
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', None)
