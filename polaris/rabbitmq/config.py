import os
from urllib.parse import urlparse

RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://localhost//')
url = urlparse(RABBITMQ_URL)

RABBITMQ_HOST = url.hostname
RABBITMQ_USERNAME = url.usename
RABBITMQ_PASSWORD = url.password
RABBITMQ_PORT = 5672
RABBITMQ_VIRTUAL_HOST = url.path[1:]
