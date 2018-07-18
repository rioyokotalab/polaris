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
    """
    A client class for parallel experiments.

    The instance of this class send a new job to workers and
    calculate next parameters in response to the requsest from a worker.

    This client adopt the RPC pattern.
    Therefore all results will be accumulated on client side.
    """

    def __init__(self, polaris):
        self.polaris = polaris
        self.exp_key = polaris.exp_key
        self.logger = polaris.logger

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

        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.channel.queue_declare(
                queue=self.request_queue_name, auto_delete=True)
        self.channel.queue_declare(queue=self.job_queue_name, auto_delete=True)

        self.callback_queue = result.method.queue
        self.channel.basic_consume(
                self.on_response, no_ack=True, queue=self.callback_queue)
        self.channel.basic_consume(
                self.on_request, no_ack=True, queue=self.request_queue_name)

    def on_request(self, ch, method, props, body):
        """
        A method to receive job request from workers
        After receiving request, this method will send a job to them.
        """

        self.send_job()

    def send_job(self):
        eval_count = self.polaris.exp_info['eval_count']
        max_evals = self.polaris.max_evals

        domain = self.polaris.domain
        trials = self.polaris.trials
        min_ei = self.polaris.min_ei
        next_params = domain.predict(trials, min_ei)

        if eval_count > max_evals or next_params is None:
            self.connection.close()

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
                exchange='',
                routing_key=self.job_queue_name,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    ),
                body=pickle.dumps(ctx)
                )

        self.polaris.exp_info['eval_count'] += 1

    def on_response(self, ch, method, props, body):
        """
        A method to receive the result of experiment from workers.
        """

        exp_payload = pickle.loads(body)
        exp_result = exp_payload['exp_result']
        params = exp_payload['params']
        exp_info = exp_payload['exp_info']

        self.polaris.trials.add(exp_result, params, exp_info)

        with open(f'{self.exp_key}.p', mode='wb') as f:
            pickle.dump(self.polaris.trials, f)

    def start(self):
        """
        A method to start consuming job requests.
        """

        self.logger.info('Start parallel execution...')

        try:
            self.channel.start_consuming()
        except (pika.exceptions.ChannelClosed, KeyboardInterrupt):
            self.logger.info('All jobs have finished')
