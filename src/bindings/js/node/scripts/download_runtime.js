const os = require('os');
const fs = require('node:fs/promises');
const nodePreGyp = require('@mapbox/node-pre-gyp');

main();

async function main() {
  const run = new nodePreGyp.Run({ argv: [] });
  const osInfo = await detectOS();
  const modulePath = run['package_json'].binary['module_path'];

  const isRuntimeDirExists = await checkDirExistence(modulePath);

  if (isRuntimeDirExists && !process.argv.includes('-f')) {
    if (process.argv.includes('--ignore-if-exists')) {
      console.error(`Directory '${modulePath}' exists, skip '--ignore-if-exists' flag passed`);
      return;
    }

    console.error(`Directory '${modulePath}' is already exist, to force runtime installation run 'npm run download_runtime -f'`);
    process.exit(1);
  }

  const originalPackageName = run['package_json'].binary['package_name'];

  let packageName = originalPackageName.replace('{letter}', osInfo.letter);
  packageName = packageName.replace('{os}', osInfo.os);

  run['package_json'].binary['package_name'] = packageName;

  run.commands.install([], () => {
    console.log('Runtime downloaded');
  });
}

async function detectOS() {
  const platform = os.platform();

  if (!['win32', 'linux'].includes(platform)) {
    console.error(`Platform '${platform}' doesn't support`);
    process.exit(1);
  }

  const platformMapping = {
    win32: {
      os: 'windows',
      letter: 'w',
    },
    linux: {
      letter: 'l',
    }
  };

  if (platform === 'linux') {
    const osReleaseData = await fs.readFile('/etc/os-release', 'utf8');
    const os = osReleaseData.includes('Ubuntu 22')
      ? 'ubuntu22'
      : osReleaseData.includes('Ubuntu 20')
      ? 'ubuntu20'
      : osReleaseData.includes('Ubuntu 18')
      ? 'ubuntu18'
      : null;

    if (!os) {
      console.error('Cannot detect your OS');
      process.exit(1);
    }

    platformMapping.linux.os = os;
  }

  return { platform, ...platformMapping[platform] };
}

async function checkDirExistence(pathToDir) {
  try {
    await fs.access(pathToDir);

    return true;
  }
  catch (err) {
    if (err.code !== 'ENOENT') throw err;

    return false;
  }
}
