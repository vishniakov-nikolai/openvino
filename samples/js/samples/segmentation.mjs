import { printShape, getArrayByImgPath, nhwc2ncwh } from '../common/index.mjs';

const modelPath =
  `../assets/models/road-segmentation-adas/road-segmentation-adas-0001.xml`;
const imgPath = '../assets/images/detroit_street_896x512.png';
const shape = [1, 3, 512, 896];

export default async function(openvinojs) {
  const model = await openvinojs.loadModel(modelPath);
  const imageArray = await getArrayByImgPath(imgPath);
  const inputTensor = new openvinojs.Tensor('f32', nhwc2ncwh(imageArray),
    shape);

  const outputTensor = await model.infer(inputTensor);

  printShape(outputTensor.shape);
}
