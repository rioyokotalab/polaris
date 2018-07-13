import copy
import importlib
import json

import pika

connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='job_queue')


def on_request(ch, method, props, body):
    ctx = json.loads(body)

    params = ctx['params']
    eval_count = ctx['eval_count']

    fn_params = copy.copy(params)
    fn_params['eval_count'] = eval_count
    fn_name = ctx['fn_name']
    fn_module = importlib.import_module(ctx['fn_module'])
    fn = getattr(fn_module, fn_name)

    exp_result = fn(fn_params)
    exp_result['eval_count'] = eval_count
    exp_result['params'] = params

    ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            body=json.dumps(exp_result)
            )
    ch.basic_ack(delivery_tag=method.delivery_tag)
    request_job()


def request_job():
    channel.basic_publish(
            exchange='',
            routing_key='request_job_queue',
            body=''
        )


if __name__ == '__main__':
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue='job_queue')

    print('Waiting for new job...')
    request_job()

    channel.start_consuming()
