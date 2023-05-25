import { printShape, getArrayByImgPath, nhwc2ncwh } from '../common/index.mjs';

const modelPath =
  '../assets/models/face-detection-0200/face-detection-0200.xml';
const imgPath = '../assets/images/peopleAndCake256x256.jpg';
const shape = [1, 3, 256, 256];

export default async function(openvinojs) {
  const model = await openvinojs.loadModel(modelPath);
  const imageArray = await getArrayByImgPath(imgPath);
  const inputTensor = new openvinojs.Tensor('f32', nhwc2ncwh(imageArray), shape);
  const outputTensor = await model.infer(inputTensor);

  printShape(outputTensor.shape);
}
