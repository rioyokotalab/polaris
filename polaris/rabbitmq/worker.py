import argparse
import copy
import importlib
import json

import pika

from polaris.rabbitmq.config import (
    RABBITMQ_HOST,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD
)


class JobWorker():
    def __init__(self, args, logger=None, debug=False):
        self.exp_key = args.exp_key
        self.use_mpi = args.mpi
        self.debug = debug

        if RABBITMQ_USERNAME and RABBITMQ_PASSWORD:
            credentials = pika.PlainCredentials(
                    RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
            rabbitmq_params = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    credentials=credentials)
        else:
            rabbitmq_params = pika.ConnectionParameters(host=RABBITMQ_HOST)
        self.connection = pika.BlockingConnection(rabbitmq_params)

        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)

            if self.debug:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

            stream = logging.StreamHandler()
            formatter = logging.Formatter(
                    '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
            stream.setFormatter(formatter)
            self.logger.addHandler(stream)
        else:
            self.logger = logger

        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='job_queue')

        self.channel.exchange_declare(
                exchange='job_exchange', exchange_type='direct')

        if self.use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def start(self):
        try:
            if (not self.use_mpi) or self.rank == 0:
                self.channel.basic_qos(prefetch_count=1)

                self.channel.queue_bind(
                        exchange='job_exchange',
                        queue='job_queue',
                        routing_key=self.exp_key)

                self.channel.basic_consume(self.on_request, queue='job_queue')

                self.logger.info('Waiting for new job...')
                self.request_job()

                self.channel.start_consuming()
            else:
                while True:
                    ctx = None
                    ctx = self.comm.bcast(ctx, root=0)
                    self.run(ctx)
        except KeyboardInterrupt:
            self.logger.info('Stop current worker...')
            self.connection.close()

            if self.use_mpi:
                from mpi4py import MPI
                MPI.Finalize()

    def on_request(self, ch, method, props, body):
        ctx = json.loads(body)
        self.logger.info(ctx)

        if self.use_mpi:
            self.comm.bcast(ctx, root=0)

        exp_result = self.run(ctx)

        ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                body=json.dumps(exp_result)
                )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        self.request_job()

    def request_job(self):
        self.channel.basic_publish(
                exchange='job_exchange',
                routing_key=f'request_{self.exp_key}',
                body=''
            )

    def run(self, ctx):
        params = ctx['params']
        eval_count = ctx['eval_count']
        fn_name = ctx['fn_name']

        fn_params = copy.copy(params)
        fn_params['eval_count'] = eval_count
        fn_module = importlib.import_module(ctx['fn_module'])
        fn = getattr(fn_module, fn_name)

        exp_result = fn(fn_params)
        exp_result['eval_count'] = eval_count
        exp_result['params'] = params

        return exp_result
