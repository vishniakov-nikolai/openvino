import os from 'node:os';
import path from 'node:path';

type SupportedTypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

interface Core {
  compileModel(
    model: Model,
    device: string,
    config?: { [option: string]: string }
  ): Promise<CompiledModel>;
  compileModelSync(
    model: Model,
    device: string,
    config?: { [option: string]: string }
  ): CompiledModel;
  readModel(modelPath: string, weightsPath?: string): Promise<Model>;
  readModel(
    modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Promise<Model>;
  readModelSync(modelPath: string, weightsPath?: string): Model;
  readModelSync(modelBuffer: Uint8Array, weightsBuffer?: Uint8Array): Model;
}
interface CoreConstructor {
  new(): Core;
}

interface Model {
  outputs: Output[];
  inputs: Output[];
  output(nameOrId?: string | number): Output;
  input(nameOrId?: string | number): Output;
  getName(): string;
}

interface CompiledModel {
  outputs: Output[];
  inputs: Output[];
  output(nameOrId?: string | number): Output;
  input(nameOrId?: string | number): Output;
  createInferRequest(): InferRequest;
}

interface Tensor {
  data: number[];
  getPrecision(): element;
  getShape(): number[];
  getData(): number[];
}
interface TensorConstructor {
  new(type: element,
      shape: number[],
      tensorData?: number[] | SupportedTypedArray): Tensor;
}

interface InferRequest {
  setTensor(name: string, tensor: Tensor): void;
  setInputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void;
  setOutputTensor(idxOrTensor: number | Tensor, tensor?: Tensor): void;
  getTensor(nameOrOutput: string | Output): Tensor;
  getInputTensor(idx?: number): Tensor;
  getOutputTensor(idx?: number): Tensor;
  getCompiledModel(): CompiledModel;
  infer(inputData?: { [inputName: string]: Tensor | SupportedTypedArray}
    | Tensor[] | SupportedTypedArray[]): { [outputName: string] : Tensor};
}

interface Output {
  anyName: string;
  shape: number[];
  toString(): string;
  getAnyName(): string;
  getShape(): number[];
  getPartialShape(): number[];
}

interface PrePostProcessor {
  build(): PrePostProcessor;
  setInputElementType(idx: number, type: element): PrePostProcessor;
  setInputModelLayout(layout: string[]): PrePostProcessor;
  setInputTensorLayout(layout: string[]): PrePostProcessor;
  preprocessResizeAlgorithm(resizeAlgorithm: resizeAlgorithm)
    : PrePostProcessor;
  setInputTensorShape(shape: number[]): PrePostProcessor;
}
interface PrePostProcessorConstructor {
  new(model: Model): PrePostProcessor;
}

declare enum element {
  u8,
  u32,
  u16,
  i8,
  i16,
  i32,
  i64,
  f32,
  f64,
}

declare enum resizeAlgorithm {
  RESIZE_NEAREST,
  RESIZE_CUBIC,
  RESIZE_LINEAR,
}

export interface NodeAddon {
  Core: CoreConstructor,
  Tensor: TensorConstructor,
  PrePostProcessor: PrePostProcessorConstructor,

  element: typeof element,
  resizeAlgorithm: typeof resizeAlgorithm,

  asyncInfer(
    InferRequest: InferRequest,
    inputData: { [inputName: string]: Tensor | SupportedTypedArray }
      | Tensor[] | SupportedTypedArray[],
    callback: (err: Error | null, inputData: Tensor[]) => void,
  ): void;
}

setPath();

export default
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../build/Release/ov_node_addon.node') as
    NodeAddon;

function setPath() {
  const { delimiter } = path;

  if (os.platform() === 'win32')
    process.env.PATH = [
      process.env.PATH,
      path.join(__dirname,
        ...'../ov_runtime/runtime/bin/intel64/Release'.split('/')),
      path.join(__dirname,
        ...'../ov_runtime/runtime/3rdparty/tbb/bin'.split('/')),
    ].join(delimiter) + delimiter;
}