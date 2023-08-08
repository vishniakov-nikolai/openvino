import util from 'node:util';
import { addon as ov, AsyncInferenceManage } from './index';

import type { Tensor } from './addon';

const modelPath = './model.xml';

const core = new ov.Core();
const model = core.readModel(modelPath);
const compiledModel = core.compileModel(model, 'CPU');
// const inferRequest = compiledModel.createInferRequest();

// inferRequest.asyncInfer(input, callback): (input) => Promise;

const inputs: { [inputName: string]: Tensor }[] = [];
// const promises = inputs.map(i => inferRequest.asyncInfer(i));



const listener = AsyncInferenceManager.process(compiledModel, inputs); // => inputs as single para

listener.on('result', (err: Error | null, result: Tensor[]) => {
  if (err) return console.log('was error:', err);

  console.log('all good:', result);
});

listener.on('finish', (_: null, results: Tensor[]) => {
  console.log('all inferences have been executed. Amount: ', results.length);
});
