const { addon: ov } = require('openvinojs-node');

const SIZE = 10000;
const TIMEOUT = 10000;

function main() {
  console.log('Start main!');

  // times(100000, createAndReleaseShape, { name: 'Shape' });
  // times(300000, createAndReleaseTensor, { name: 'Tensor' });
  // times(1000000, createAndReleasePPP(), { name: 'PrePostProcessor' });
  times(300000, createAndReleaseCore, { name: 'Core' }); // FIXME: leaks!
  // times(2000, createAndReleaseModel(), { name: 'Model' }); // FIXME: leaks!
  // times(100, createAndReleaseCompiledModel(), { name: 'CompiledModel' }); // FIXME: leaks!
  // times(500000, createAndReleaseModelInputs(), { name: 'Model inputs' }); //  FIXME: leaks!

  console.log('Done main!');
}

setTimeout(() => main(), TIMEOUT);
setTimeout(() => main(), TIMEOUT*2);

function createAndReleaseCore() {
  const core = new ov.Core();
}

function createAndReleaseTensor() {
  const largeArray = new Float32Array(SIZE).fill(1);

  new ov.Tensor(ov.element.f32, [1, SIZE], largeArray);
}

function createAndReleaseShape() {
  const largeArray = new Uint32Array(SIZE).fill(1);

  new ov.Shape(SIZE, largeArray);
}

function createAndReleasePPP() {
  const core = new ov.Core();
  const model = core.readModel('../../assets/models/classification.xml');

  return () => new ov.PrePostProcessor(model).build();
}

function createAndReleaseModel() {
  const core = new ov.Core();

  return () => core.readModel('../../assets/models/classification.xml');
}

function createAndReleaseCompiledModel() {
  const core = new ov.Core();
  const model = core.readModel('../../assets/models/classification.xml');

  return () => core.compileModel(model, 'AUTO');
}

function createAndReleaseModelInputs() {
  const core = new ov.Core();
  const model = core.readModel('../../assets/models/classification.xml');
  const compiledModel = core.compileModel(model, 'CPU');

  return () => compiledModel.inputs;
}

function times(number, fn, { name } = {}) {
  const transformedName = name ? `'${name}' ` : '';
  console.log(`Start ${transformedName} ${number} times`);

  for (let i = 0; i < number; i++) {
    fn();
  }

  console.log(`Done ${transformedName}`);
}
