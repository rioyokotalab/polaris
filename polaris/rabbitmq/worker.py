import argparse
import copy
import importlib
import json

from mpi4py import MPI
import pika

from polaris.rabbitmq.config import (
    RABBITMQ_HOST,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD
)


class JobWorker():
    def __init__(self, args):
        self.exp_key = args.exp_key
        self.use_mpi = args.mpi

        if RABBITMQ_USERNAME and RABBITMQ_PASSWORD:
            credentials = pika.PlainCredentials(
                    RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
            rabbitmq_params = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    credentials=credentials)
        else:
            rabbitmq_params = pika.ConnectionParameters(host=RABBITMQ_HOST)
        self.connection = pika.BlockingConnection(rabbitmq_params)

        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='job_queue')

        self.channel.exchange_declare(
                exchange='job_exchange', exchange_type='direct')

        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def start(self):
        try:
            if (not self.use_mpi) or self.rank == 0:
                self.channel.basic_qos(prefetch_count=1)

                if self.exp_key:
                    self.channel.queue_bind(
                            exchange='job_exchange',
                            queue='job_queue',
                            routing_key=self.exp_key)

                self.channel.basic_consume(self.on_request, queue='job_queue')

                print('Waiting for new job...')
                self.request_job()

                self.channel.start_consuming()
            else:
                while True:
                    ctx = None
                    ctx = self.comm.bcast(ctx, root=0)
                    self.run(ctx)
        except KeyboardInterrupt:
            print('Stop current worker...')
            self.connection.close()

            if self.use_mpi:
                MPI.Finalize()

    def on_request(self, ch, method, props, body):
        ctx = json.loads(body)

        print(ctx)

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
                exchange='',
                routing_key='request_job_queue',
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Start a job worker for polaris')
    parser.add_argument('--exp-key', '--e')
    parser.add_argument('--mpi', '--m', action='store_true')
    args = parser.parse_args()

    worker = JobWorker(args)
    worker.start()
