import addon from './addon';
import util from 'node:util';

import type { Tensor, CompiledModel } from './addon';

type asyncCallback = (err: Error | null, result: Tensor[] | null) => void;
type finishCallback =
  (err: null, results: PromiseSettledResult<Tensor[]>[]) => void;
type SupportedEvents = 'result' | 'finish';

class AsyncInferenceManager {
  static process(compiledModel: CompiledModel, inputs: { [inputName: string]: Tensor }[], timeout = 10_000) {
    const inferRequest = compiledModel.createInferRequest();

    const list = inputs.map(i => {
      const currentTime = Date();

      return util.promisify(inferRequest.asyncInfer)(i, currentTime + timeout);
    });

    const subscribers: {
      result?: asyncCallback[],
      finish?: finishCallback[],
    } = {};

    list.forEach(l => l.then(
      (result: Tensor[]) => onResult(null, result),
      (err: Error) => onResult(err),
    ));
    Promise.allSettled(list).then(onFinish);

    return { on: eventProcessor };

    function eventProcessor(
      event: SupportedEvents,
      callback: asyncCallback | finishCallback
    ) {
      if (!Array.isArray(subscribers[event])) subscribers[event] = [];

      if (event === 'result')
        subscribers[event]?.push(callback as asyncCallback);
      if (event === 'finish')
        subscribers[event]?.push(callback as finishCallback);
    }

    function onResult(err: Error | null, result?: Tensor[]) {
      const eventName = 'result';
      const listeners = subscribers[eventName] || [];

      listeners.forEach(l => l(err, result || null));
    }

    function onFinish(results: PromiseSettledResult<Tensor[]>[]) {
      const eventName = 'finish';
      const listeners = subscribers[eventName] || [];

      listeners.forEach(l => l(null, results));
    }
  }
}

fn(...parameters, callback) {
  // ...

  try {
    const ref = Object.callback(null, result);
  } catch(e) {
    callback(e);
  }
}

Object.cancel(ref);


export { addon, AsyncInferenceManager };
