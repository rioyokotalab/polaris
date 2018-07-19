=====================
Example
=====================

In default, your training function (ex: pseudo_train) was called with *params* and *exp_info* arguments.
You also can pass the extra arguments as tuple with args keyword for Polaris instance (picklable instance only).

Single Execution
=====================

.. code-block:: python

  from polaris import Polaris, STATUS_SUCCESS, Trials, Bounds


  def pseudo_train(params, exp_info, extra_args):
      lr_squared = (params['lr'] - 0.006) ** 2
      weight_decay_squared = (params['weight_decay'] - 0.02) ** 2
      loss = lr_squared + weight_decay_squared

      return {
          'loss':  loss,
          'status': STATUS_SUCCESS,
      }


  if __name__ == '__main__':
      bounds = [
          Bounds('lr', 0.001, 0.01),
          Bounds('weight_decay', 0.0002, 0.04),
      ]
      trials = Trials()

      extra_args = 'extra_args!!'

      polaris = Polaris(
              pseudo_train, bounds, 'bayesian_opt',
              trials, max_evals=100, exp_key='this_is_test', args=(extra_args,))
      best_params = polaris.run()
      print(best_params)

Parallel Execution
=====================

Single Process
---------------------

#. Run `rabbitmq-server`
#. Set `RABBITMQ_URL` environment variable (Ex: ampq://guest:guest@localhost//)
#. Run `polaris-worker --exp-key this_is_test`
#. Run client as follows

Multiple Processes (Use MPI)
-----------------------------

#. Run `rabbitmq-server`
#. Run `pip install mpi4py`
#. Set `RABBITMQ_URL` environment variable (Ex: ampq://guest:guest@localhost//)
#. Run `mpirun -n 4 polaris-worker --mpi --exp-key this_is_test`
#. Run client as follows


.. code-block:: python

  ...

  if __name__ == '__main__':
      bounds = [
          Bounds('lr', 0.001, 0.01),
          Bounds('weight_decay', 0.0002, 0.04),
      ]
      trials = Trials()
      polaris = Polaris(
              pseudo_train, bounds, 'bayesian_opt',
              trials, max_evals=100, exp_key='this_is_test')
      best_params = polaris.run_parallel()
      print(best_params)
