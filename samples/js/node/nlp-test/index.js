main();

async function main() {
  const { pipeline } = await import('@xenova/transformers');

  let pipe = await pipeline(
    'text-generation',
    'helenai/gpt2-ov',
    { 'model_file_name': 'openvino_model.xml' },
  );

  let out = await pipe('I love transformers!');

  console.log(out);
}
