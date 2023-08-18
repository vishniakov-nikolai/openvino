const { addon: ov } = require('openvinojs-node');

const modelXMLPath =
  '../assets/models/codet5p-220m-py/openvino_encoder_model.xml';

const core = new ov.Core();
const model = core.readModel(modelXMLPath);
new ov.PrePostProcessor(model)
  .setInputElementType(0, ov.element.i32)
  .setInputElementType(1, ov.element.i32)
  .build();
const compiledModel = core.compileModel(model, 'CPU');

const inferRequest = compiledModel.createInferRequest();

const maxId = 32000;
const seqSize = 900;
const tensorsNumbers = 1000;

const attentionMask =
  new ov.Tensor(ov.element.i32, [1, seqSize], new Int32Array(seqSize).fill(1));

const warmUpInput = getVectors(3, seqSize);
const inputData = getVectors(tensorsNumbers, seqSize);

for (let i of warmUpInput)
  inferRequest.infer({ 'attention_mask': attentionMask, 'input_ids': i });

// console.time('s');
const st = process.hrtime();
for (let i of inputData) {
  // console.log(i.data);
  inferRequest.infer({ 'attention_mask': attentionMask, 'input_ids': i });
}
const rt = process.hrtime(st);
// console.timeEnd('s');

const timeInSec = rt[0] + rt[1]/1000000000;

// console.log(inferRequest.getOutputTensor().data[10]);
console.log(`Seq. length:\t${seqSize}`);
console.log(`Inferences:\t${tensorsNumbers}`);
console.log(`Full:\t\t${(timeInSec).toFixed(2)}s`);
console.log(`Average:\t${(timeInSec/tensorsNumbers).toFixed(2)}s`);

function getRandomVector(length) {
  return new ov.Tensor(ov.element.i32, [1, length],
    new Int32Array(rand(length)));
}

function getVectors(count, length) {
  return (new Array(count)).fill(0).map(() => getRandomVector(length));
}

function rand(count) {
  const arr = new Array(count);

  for (let i = 0; i < count; i++) arr[i] = Math.floor(Math.random()*maxId);

  return arr;
}
