const fs = require('node:fs');
const { addon: ov } = require('openvino-node');

const { cv } = require('opencv-wasm');
const { getImageData } = require('../helpers.js');

// Parsing and validation of input arguments
if (process.argv.length !== 5)
  throw new Error(`Usage: ${process.argv[1]} <path_to_model> `
    + '<path_to_image> <device_name>');

const modelPath = process.argv[2];
const imagePath = process.argv[3];
const deviceName = process.argv[4];

main(modelPath, imagePath, deviceName);

async function main(modelPath, imagePath, deviceName) {
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  const model = await core.readModel(modelPath);

  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  if (model.outputs.length !== 1)
    throw new Error('Sample supports only single output topologies');

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imgData = await getImageData(imagePath);

  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);

  const tensorData = new Float32Array(image.data);
  const shape = [1, image.rows, image.cols, 3];
  const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

  //----------------- Step 4. Apply preprocessing ------------------------------
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().tensor().setShape(shape).setLayout('NHWC');
  _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);
  _ppp.input().model().setLayout('NHWC');
  _ppp.output().tensor().setElementType(ov.element.f32);
  _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, deviceName);

  //---------------- Step 6. Create infer request and do inference asynchronously
  const inferRequest = compiledModel.createInferRequest();

  console.log('== Start Inference')
  inferRequest.inferAsync([inputTensor]).then(() => {
    console.log('== Inference Done')
    //----------------- Step 7. Process output -----------------------------------
    const outputLayer = compiledModel.outputs[0];
    const resultInfer = inferRequest.getTensor(outputLayer);
    const resultIndex = resultInfer.data.indexOf(Math.max(...resultInfer.data));

    console.log("== Result ==");
    console.log(`  Index: ${resultIndex}`);

    const imagenetClassesMapContent = fs.readFileSync('../../assets/datasets/imagenet_class_index.json', 'utf-8');
    const imagenetClassesMap = JSON.parse(imagenetClassesMapContent);
    const imagenetClasses = ['background', ...Object.values(imagenetClassesMap)];

    console.log(`  Label: ${imagenetClasses[resultIndex][1]}`);
    console.log("============");
  });

  for (let i = 0; i < 10; i++) {
    console.log(`= is alive ${i}`);
    await sleep(0);
  }
}

function sleep(timeoutMs) {
  return new Promise((res, rej) => setTimeout(() => res(), timeoutMs));
}
