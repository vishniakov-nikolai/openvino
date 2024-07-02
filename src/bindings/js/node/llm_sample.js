const { addon: ov } = require('.');

const DEVICE = 'CPU';

class Generate {
  constructor(model, tokenizer, detokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.detokenizer = detokenizer;
  }

  // Returns BigInt64Array
  async tokenize(str) {
    const ir = this.tokenizer.createInferRequest();
    const tensor = new ov.Tensor([str]);

    const result = await ir.inferAsync({ 'string_input': tensor });

    return result['input_ids'].data;
  }

  async detokenize(bigIntArr) {
    const detokenizerInput = this.detokenizer.inputs[0];
    const ir = this.detokenizer.createInferRequest();
    const tensor = new ov.Tensor('i64', [1, bigIntArr.length], bigIntArr);
    const result = await ir.inferAsync({ [detokenizerInput.anyName]: tensor });

    return result['string_output'].data;
  }

  async run(str, _opt) {
    const opt = Object.assign({
      max: 100,
      callback: async (token) => {
        const symbol = await this.detokenize(new BigInt64Array([BigInt(token)]));

        console.log(symbol);
      },
    }, _opt || {});

    let tokens = await this.tokenize(str);

    for (let i = 0; i < opt.max; i++) {
      const length = tokens.length;
      const input = {
        'input_ids': new ov.Tensor('i64', [1, length], tokens),
        'attention_mask': new ov.Tensor('i64', [1, length], (new BigInt64Array(length)).fill(BigInt(1))),
        'position_ids': new ov.Tensor('i64', [1, length], populatePositionIdx(length)),
        'beam_idx': new ov.Tensor('i32', [1], new Int32Array([0])),
      };

      const ir = this.model.createInferRequest();
      const { logits } = await ir.inferAsync(input);

      const shape = logits.getShape();
      const data = logits.data.slice(-shape[shape.length - 1]);

      const index = argmax(data);

      await opt.callback(index);

      const newTokens = new BigInt64Array(tokens.length + 1);
      newTokens.set(tokens);
      newTokens[newTokens.length - 1] = BigInt(index);

      tokens = newTokens;
    }

    return tokens;
  }
}

// const testStr = 'Write me a function to calculate the first 10 digits of the fibonacci sequence in Python and print it out to the CLI.';
// const testStr = '[INST] Hello, how are you? [/INST]I\'m doing great. How can I help you today?</s> [INST] I\'d like to show off how chat templating works! [/INST]';
const testStr = 'Say "Hello" in German';

main(testStr, 250);

async function main(input, max, showOutput) {
  const core = new ov.Core();

  core.addExtension('./bin/libopenvino_tokenizers.so')

  const model = await core.readModel('/home/nvishnya/Code/TinyLlama-1.1B-Chat-v1.0/out/openvino_model.xml');
  const tokenizerModel = await core.readModel('/home/nvishnya/Code/openvino_tokenizers/output_dir/openvino_tokenizer.xml');
  const detokenizerModel = await core.readModel('/home/nvishnya/Code/openvino_tokenizers/output_dir/openvino_detokenizer.xml');

  const compiledModel = await core.compileModel(model, DEVICE);
  const compiledTokenizer = await core.compileModel(tokenizerModel, DEVICE);
  const compiledDetokenizer = await core.compileModel(detokenizerModel, DEVICE);

  const g = new Generate(compiledModel, compiledTokenizer, compiledDetokenizer);

  const intervalId = setInterval(() => console.log(`tick: ${Math.floor(new Date().getTime()/1000)}`), 1000);

  let acc = '';
  process.stdout.write(`Output: ...`);

  const label = `Generation of ${max} tokens spends`;
  console.time(label);
  const tokens = await g.run(input, {
    max,
    callback: async (token) => {
      if (!showOutput) return;

      const symbol = await g.detokenize(new BigInt64Array([BigInt(token)]));

      acc = `${acc} ${symbol}`;

      process.stdout.clearLine(0);
      process.stdout.cursorTo(0);
      process.stdout.write(`Output: "${acc}"...`);
    },
  });
  console.timeEnd(label);

  if (showOutput) {
    const num = getShifts(acc);
    for (let i = num; i > 0; i--) process.stdout.clearLine(i);
    process.stdout.cursorTo(0);
    process.stdout.write(`Output: "${acc}"\nDone.`);
  }

  const outStr = await g.detokenize(tokens);
  console.log(`\n${outStr}`);

  clearInterval(intervalId);
}

function populatePositionIdx(length) {
  const arr = new BigInt64Array(length);

  for (let i = 0; i < length; i++) arr[i] = BigInt(i);

  return arr;
}

function argmax(logits) {
  if (logits.length === 0)
    throw new Error('The logits array should not be empty');

  let maxIndex = 0;
  let maxValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxValue) {
      maxValue = logits[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}
