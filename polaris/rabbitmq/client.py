import json
import random
import uuid

import pika


class JobClient():

    def __init__(self, polaris):
        self.polaris = polaris
        self.eval_count = 0

        self.connection = pika.BlockingConnection(
               pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='request_job_queue')
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
                self.on_response, no_ack=True, queue=self.callback_queue)
        self.channel.basic_consume(
                self.on_request, no_ack=True, queue='request_job_queue')

    def on_request(self, ch, method, props, body):
        if self.eval_count >= self.polaris.max_evals:
            self.channel.close()

        domain = self.polaris.domain
        trials = self.polaris.trials
        next_params = domain.predict(trials)

        fn = self.polaris.fn
        fn_module = fn.__module__
        fn_name = fn.__name__
        ctx = {
            'eval_count': self.eval_count,
            'params': next_params,
            'fn_module': fn_module,
            'fn_name': fn_name,
        }

        self.eval_count += 1
        self.channel.basic_publish(
                exchange='',
                routing_key='job_queue',
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

    def run(self):
        print('Start parallel execution...')
        self.channel.start_consuming()
