import json

import pika

from polaris.rabbitmq.config import (
    RABBITMQ_HOST,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD
)


class JobClient():

    def __init__(self, polaris):
        self.polaris = polaris
        self.eval_count = 0

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
        result = self.channel.queue_declare(exclusive=True)
        self.channel.queue_declare(queue='job_queue')
        self.channel.queue_declare(queue='request_job_queue')

        self.callback_queue = result.method.queue
        self.channel.basic_consume(
                self.on_response, no_ack=True, queue=self.callback_queue)
        self.channel.basic_consume(
                self.on_request, no_ack=True, queue='request_job_queue')
        self.channel.exchange_declare(
                exchange='job_exchange', exchange_type='direct')

    def on_request(self, ch, method, props, body):
        self.send_job()

    def send_job(self):
        if self.eval_count >= self.polaris.max_evals:
            self.connection.close()

        domain = self.polaris.domain
        trials = self.polaris.trials
        next_params = domain.predict(trials)

        fn = self.polaris.fn
        fn_module = fn.__module__
        fn_name = fn.__name__
        if fn_module == '__main__':
            # TODO This transformation is ugly...
            import __main__
            fn_module = __main__.__file__.replace('/', '.').replace('.py', '')

        self.eval_count += 1
        ctx = {
            'eval_count': self.eval_count,
            'params': next_params,
            'fn_module': fn_module,
            'fn_name': fn_name,
        }

        self.channel.basic_publish(
                exchange='job_exchange',
                routing_key=trials.exp_key,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    ),
                body=json.dumps(ctx)
                )

    def on_response(self, ch, method, props, body):
        exp_result = json.loads(body)
        params = exp_result['params']
        eval_count = exp_result['eval_count']

        del exp_result['params']
        del exp_result['eval_count']

        self.polaris.trials.add(exp_result, params, eval_count)

    def start(self):
        print('Start parallel execution...')

        try:
            self.send_job()
            self.channel.start_consuming()
        except pika.exceptions.ChannelClosed:
            print('All jobs have finished')
