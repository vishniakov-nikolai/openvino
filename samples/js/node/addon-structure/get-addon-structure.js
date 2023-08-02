const { addon: ov } = require('openvinojs-node');
const fs = require('node:fs');

const addonStructure = getAddonStructure();

fs.writeFileSync('addon-structure.json',
  JSON.stringify(addonStructure, null, 2));

function getAddonStructure() {
  const core = new ov.Core();
  const model = core.readModel('../../assets/models/classification.xml');
  const compiledModel = core.compileModel(model, 'CPU');
  const tensor = new ov.Tensor(ov.element.f32, [1], new Float32Array([1]));
  const shape = new ov.Shape(2, new Uint32Array([1, 2]));
  const inferRequest = compiledModel.createInferRequest();

  const input = compiledModel.inputs[0];
  const output = compiledModel.outputs[0];

  console.log(input);

  const ppp = new ov.PrePostProcessor(model);

  const ovStructure = printObj('ov (addon)', ov, false);
  const coreStructure = printObj('Core', core);
  const modelStructure = printObj('Model', model);
  const compiledModelStructure = printObj('CompiledModel', compiledModel);
  const tensorStructure = printObj('Tensor', tensor);
  const shapeStructure = printObj('Shape', shape);
  const inferRequestStructure = printObj('InferRequest', inferRequest);
  const inputStructure = printObj('Input', input);
  const outputStructure = printObj('Output', output);
  const pppStructure = printObj('PrePostProcessor', ppp);

  const elementStructure = printObj('ov.element', ov.element, false);
  const resizeAlgorithmStructure = printObj('ov.resizeAlgorithms',
    ov.resizeAlgorithms, false);

  return [
    ovStructure,
    coreStructure,
    modelStructure,
    compiledModelStructure,
    tensorStructure,
    shapeStructure,
    inferRequestStructure,
    inputStructure,
    outputStructure,
    pppStructure,
    elementStructure,
    resizeAlgorithmStructure,
  ];
}

function getObjectStructure(name, obj, isProto = true) {
  const structure = { name, attributes: [], methods: [] };
  const innerObj = isProto ? Object.getPrototypeOf(obj) : obj;

  const props = Object.getOwnPropertyNames(innerObj);
  props.forEach(p =>
    typeof obj[p] === 'function'
      ? structure.methods.push(p)
      : structure.attributes.push(p),
  );

  return structure;
}

function printObj(objName, obj, isProto = true) {
  const structure = getObjectStructure(objName, obj, isProto);
  const { name, attributes, methods } = structure;

  console.log(`${name}:`);
  attributes.reverse().forEach(a => console.log(`  ${a}`));
  methods.reverse().forEach(m => console.log(`  ${m}()`));

  if (attributes.length || methods.length) console.log();

  return structure;
}
