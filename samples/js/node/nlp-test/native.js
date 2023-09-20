main();

async function main() {
  const transformers = await import('@xenova/transformers');
  const { pipeline, Model } = transformers;

  console.log(Model)

  let classifier = await pipeline('sentiment-analysis', {});
  return;
  let result = await classifier('I hate transformers!');

  console.log(result);
}
