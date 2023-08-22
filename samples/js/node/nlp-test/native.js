main();

async function main() {
  const { pipeline } = await import('@xenova/transformers');

  const gen = await pipeline('text-generation');
  const out = await gen('Hello, I\'m a language model');

  console.log(out);
}
