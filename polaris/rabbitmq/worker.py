import copy
import importlib
import pickle

import pika

from polaris.rabbitmq.config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_VIRTUAL_HOST,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD,
)


class JobWorker():
    """
    A worker class for parallel experiments.

    You can start the worker like below.
    `polaris-worker --exp-key this_is_test`

    And if you want to run this worker on multi node environment,
    you have to add `--mpi` flag.
    """

    def __init__(self, args, logger=None, debug=False):
        self.exp_key = args.exp_key
        self.use_mpi = args.mpi
        self.debug = debug

        self.job_queue_name = f'job_{self.exp_key}'
        self.request_queue_name = f'request_{self.exp_key}'

        if RABBITMQ_USERNAME and RABBITMQ_PASSWORD:
            credentials = pika.PlainCredentials(
                    RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
            rabbitmq_params = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    virtual_host=RABBITMQ_VIRTUAL_HOST,
                    credentials=credentials)
        else:
            rabbitmq_params = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    virtual_host=RABBITMQ_VIRTUAL_HOST)

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
        self.channel.queue_declare(
                queue=self.request_queue_name, auto_delete=True)
        self.channel.queue_declare(queue=self.job_queue_name, auto_delete=True)

        if self.use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def start(self):
        try:
            if (not self.use_mpi) or self.rank == 0:
                self.channel.basic_qos(prefetch_count=1)

                self.channel.basic_consume(
                        self.on_request, queue=self.job_queue_name)

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
        ctx = pickle.loads(body)
        self.logger.info(ctx)

        if self.use_mpi:
            self.comm.bcast(ctx, root=0)

        exp_result = self.run(ctx)
        exp_payload = {
            'exp_result': exp_result,
            'params': ctx['params'],
            'exp_info': ctx['exp_info'],
        }

        ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            body=pickle.dumps(exp_payload)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        self.request_job()

    def request_job(self):
        self.channel.basic_publish(
                exchange='',
                routing_key=self.request_queue_name,
                body=''
            )

    def run(self, ctx):
        params = ctx['params']
        exp_info = ctx['exp_info']
        fn_module = ctx['fn_module']
        fn_name = ctx['fn_name']
        args = ctx['args']

        fn_params = copy.copy(params)
        fn_module = importlib.import_module(fn_module)
        fn = getattr(fn_module, fn_name)

        if args is not None:
            exp_result = fn(fn_params, exp_info, *args)
        else:
            exp_result = fn(fn_params, exp_info)

        return exp_result
