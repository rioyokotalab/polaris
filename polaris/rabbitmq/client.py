import pickle

import pika

from polaris.rabbitmq.config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_VIRTUAL_HOST,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD,
)


class JobClient():

    def __init__(self, polaris):
        self.polaris = polaris
        self.exp_key = polaris.exp_key
        self.logger = polaris.logger

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

        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.channel.queue_declare(queue='job_queue')
        self.channel.queue_declare(queue='request_job_queue')

        self.channel.exchange_declare(
                exchange='job_exchange', exchange_type='direct')
        self.channel.queue_bind(
                exchange='job_exchange',
                queue='request_job_queue',
                routing_key=f'request_{self.exp_key}')
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
                self.on_response, no_ack=True, queue=self.callback_queue)
        self.channel.basic_consume(
                self.on_request, no_ack=True, queue='request_job_queue')

    def on_request(self, ch, method, props, body):
        self.send_job()

    def send_job(self):
        eval_count = self.polaris.exp_info['eval_count']
        max_evals = self.polaris.max_evals

        if eval_count > max_evals:
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

        ctx = {
            'fn_module': fn_module,
            'fn_name': fn_name,
            'params': next_params,
            'exp_info': self.polaris.exp_info,
            'args': self.polaris.args,
        }

        self.channel.basic_publish(
                exchange='job_exchange',
                routing_key=self.exp_key,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    ),
                body=pickle.dumps(ctx)
                )

        self.polaris.exp_info['eval_count'] += 1

    def on_response(self, ch, method, props, body):
        exp_payload = pickle.loads(body)
        exp_result = exp_payload['exp_result']
        params = exp_payload['params']
        exp_info = exp_payload['exp_info']

        self.polaris.trials.add(exp_result, params, exp_info)

    def start(self):
        self.logger.info('Start parallel execution...')

        try:
            self.send_job()
            self.channel.start_consuming()
        except pika.exceptions.ChannelClosed:
            self.logger.info('All jobs have finished')
