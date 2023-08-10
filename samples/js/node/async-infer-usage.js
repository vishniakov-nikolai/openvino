const { addon: ov } = require('openvinojs-node');
const util = require('node:util');

const cv2 = require('opencv.js');
const { getImageData } = require('./helpers.js');

main();

async function main() {
  const modelPath = '../assets/models/v3-small_224_1.0_float.xml';
  const inputData = [];
  // const inputData = [
  //   Float32Array.from({length: 150528}, () => Math.random() ),
  //   Float32Array.from({length: 150528}, () => Math.random() ),
  //   Float32Array.from({length: 150528}, () => Math.random() ),
  // ];
  // e.g. [tensorData1: TypedArray, tensorData2: TypedArray, ...]

  const imagesPaths = [
    '../assets/images/coco.jpg',
    '../assets/images/shih_tzu.jpg',
  ];

  for (let i = 0; i < imagesPaths.length; i++) {
    inputData.push(await getArrayByImgPath(imagesPaths[i]));
  }

  // return console.log('== Done');

  const core = new ov.Core();
  const model = core.readModel(modelPath);
  const compiledModel = core.compileModel(model, 'CPU');

  const inferRequest = compiledModel.createInferRequest();

  const promises = inputData.map(i => {
    // : asyncInfer({ inferRequest: InferRequest, [inputName: string]: Tensor })
    // => Promise<{ [outputName: string]: Tensor }>
    const promisifiedAsyncInfer = util.promisify(ov.asyncInfer);

    // : Promise<{ [outputName: string]: Tensor }>

    console.log(i[0]);

    return promisifiedAsyncInfer(inferRequest, [i]);
  });

  Promise.all(promises).then(outputs => {
    console.log('== Done', outputs.length);

    for (const i in outputs) {
      const data = outputs[i]['MobilenetV3/Predictions/Softmax:0'].getData();

      logResult(data);
    }
  }).catch((err) => {
    console.log('Error: ', err);
  });
}

async function getArrayByImgPath(imagePath) {
  const imgData = await getImageData(imagePath);

  // Use OpenCV.js to preprocess image.
  const originalImage = cv2.matFromImageData(imgData);
  const image = new cv2.Mat();
  // The MobileNet model expects images in RGB format.
  cv2.cvtColor(originalImage, image, cv2.COLOR_RGBA2RGB);
  cv2.resize(image, image, new cv2.Size(224, 224));

  return new Float32Array(image.data);
}

function logResult(data) {
  const predictions = Array.from(data)
    .map((prediction, classId) => ({ prediction, classId }))
    .sort(({ prediction: predictionA }, { prediction: predictionB }) =>
      predictionA === predictionB ? 0 : predictionA > predictionB ? -1 : 1);

  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}`),
  );
}
