import { printShape, printClass, getArrayByImgPath } from '../common/index.mjs';

const modelPath = `../assets/models/v3-small_224_1.0_float.xml`;
const imgPath = '../assets/images/coco224x224.jpg';
const shape = [1, 224, 224, 3];
// const layout = 'NHWC';

export default async function(openvinojs) {
  const model = await openvinojs.loadModel(modelPath);
  const imageArray = await getArrayByImgPath(imgPath);
  const inputTensor = new openvinojs.Tensor('f32', imageArray, shape);

  const outputTensor = await model.infer(inputTensor);

  printShape(outputTensor.shape);
  await printClass(outputTensor);
}
