const os = require('os');
const path = require('path');
const fs = require('node:fs/promises');
const decompress = require('decompress');

const packageJson = require('../package.json');

/**
 * Object specifies archeive name and path to dependencies list in it.
 * 
 * @typedef {Object} OsSpecificFiles
 * @property {string} archive
 * @property {string[]} dependencies
 */

/**
 * Object contains path to binaries info.
 * 
 * @typedef {Object} OvBinariesInfo
 * @property {string} version
 * @property {string} url
 * @property {OsSpecificFiles} linux
 * @property {OsSpecificFiles} windows
 * @property {OsSpecificFiles} macos
 */

/**
 * Object contains arch and OS info.
 * 
 * @typedef {Object} SystemInfo
 * @property {string} os
 * @property {string} arch
 */

const BIN_PATH = './bin';

const codeENOENT = 'ENOENT';

class RuntimeExistsError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RuntimeExistsError';
    Error.captureStackTrace(this, RuntimeExistsError);
  }
}

if (require.main === module) {
  main();
}

async function main() {
  const isForceRun = process.argv.includes('-f');
  const ignoreIfExists = process.argv.includes('--ignore-if-exists');

  try {
    const exists = await checkIfDirectoryExists(BIN_PATH);

    if (exists && ignoreIfExists) {
      console.warn(
        `Directory '${BIN_PATH}' already exists. Skipping runtime downloading because 'ignoreIfExists' flag is passed.`
      );
      return;
    }

    if (exists && !isForceRun) {
      throw new RuntimeExistsError(
        `Directory '${BIN_PATH}' already exists. To force runtime downloading use 'force' flag.`
      );
    }

    await removeDirectory(BIN_PATH);
    await fs.mkdir(BIN_PATH);
  } catch(error) {
    if (error instanceof RuntimeExistsError) {
      console.error(
        `Directory '${BIN_PATH}' already exists. ` + 
        'To force runtime downloading run \'npm run download_binaries -- -f\''
      );
    } else {
      throw error;
    }
    process.exit(1);
  }

  const systemInfo = getArchAndOs();

  const ovRuntimeUrl = getOvRuntimeUrl(packageJson.binary, systemInfo);
  const bindingsUrl = getOvjsBindingsUrl(packageJson.binary, systemInfo);

  const { dependencies } = packageJson.binary[systemInfo.os];

  await fetchFiles(ovRuntimeUrl, BIN_PATH, dependencies);
  await fetchFiles(bindingsUrl, BIN_PATH, 'ov_node_addon.node');
}

/**
 * Download archive by URL, uncompress to tmp dir and copy listed files
 *
 * @async
 * @function fetchFiles
 * @param {string} url - URL to archive.
 * @param {string} destinationPath - Where put files.
 * @param {string[]|string} _paths - File paths to copy.
 */
async function fetchFiles(url, destinationPath, _paths) {
  const paths = typeof _paths === 'string' ? [_paths] : _paths;
  const tmpDir = 
    path.join(os.tmpdir(), `ovjs-binaries-${(new Date()).getTime()}`);
  console.log(`Start downloading: ${url}`)
  const response = await fetch(url);
  const buffer = await readResponse(response, (progress) => 
    console.log(`Downloaded: ${progress}%`));
  console.log('Downloaded');
  await decompress(buffer, tmpDir, { strip: 1 });
  
  for (const filePath of paths) {
    const filename = path.basename(filePath);
    const dest = path.join(destinationPath, filename);

    await fs.copyFile(path.join(tmpDir, filePath), dest);
    console.log(`Copied: ${dest}`);
  }
  await removeDirectory(tmpDir);
  console.log('Done');
}

/**
 * Get system info.
 *
 * @function getArchAndOs
 * @returns {SystemInfo}
 */
function getArchAndOs() {
  const osMapping = {
    win32: 'windows',
    linux: 'linux',
    darwin: 'macos',
  };
  const archMapping = {
    arm64: 'arm64',
    x64: 'x86_64',
  };

  const supportedPlatforms = Object.keys(osMapping);
  const supportedArchs = Object.keys(archMapping);

  const platform = os.platform();
  const arch = os.arch();

  if (!supportedArchs.includes(arch)) {
    throw new Error(`Architecture '${arch}' is not supported.`);
  }

  if (!supportedPlatforms.includes(platform)) {
    throw new Error(`Platform '${platform}' is not supported.`);
  }

  return {
    arch: archMapping[arch],
    os: osMapping[platform],
  };
}

/**
 * Check if directory exists.
 *
 * @async
 * @function checkIfDirectoryExists
 * @param {string} directoryPath - The directory path.
 * @returns {Promise<boolean>}
 */
async function checkIfDirectoryExists(directoryPath) {
  try {
    await fs.access(directoryPath);
    return true;
  } catch (error) {
    if (error.code === codeENOENT) {
      return false;
    }
    throw error;
  }
}

/**
 * Get full URL to OpenVINO runtime archive.
 *
 * @function getRuntimeArchiveUrl
 * @param {OvBinariesInfo} binariesInfo 
 * @param {SystemInfo} systemInfo
 * @returns {string}
 */
function getOvRuntimeUrl(binariesInfo, systemInfo) {
  const { url, version } = binariesInfo;
  const { os } = systemInfo;
  const archiveName = binariesInfo[os].archive;

  return new URL(path.join(url, version, os, archiveName)).toString();
}
/**
 * Get full URL to OpenVINO JavaScript bindings.
 *
 * @function getOvjsBindingsUrl
 * @param {OvBinariesInfo} binariesInfo 
 * @param {SystemInfo} systemInfo
 * @returns {string}
 */
function getOvjsBindingsUrl(binariesInfo, systemInfo) {
  const { artifactsUrl, version, bindings } = binariesInfo;
  const { os } = systemInfo;

  return new URL(path.join(artifactsUrl, version, os, bindings)).toString();
}

/**
 * Remove directory and its content.
 *
 * @async
 * @function removeDirectory
 * @param {string} path - The directory path.
 * @returns {Promise<void>}
 */
async function removeDirectory(path) {
  try {
    await fs.rm(path, { recursive: true, force: true });
  } catch (error) {
    if (error.code !== codeENOENT) {
      throw error;
    }
  }
}

/**
 * Read Response object.
 * 
 * @async
 * @param {Response} response Response object to read.
 * @param {function} progressCallback The function to call with progress updates.
 * @returns {Promise<Buffer>} Promise resolves Buffer.
 */
async function readResponse(response, progressCallback) {
  const total = parseInt(response.headers.get('Content-Length') ?? 0);
  const reader = response.body.getReader();

  let loaded = 0;
  let buffer = new Uint8Array(total);

  await read();

  return Buffer.from(buffer);

  async function read() {
    const { done, value } = await reader.read();

    if (done) return;

    const curLoaded = loaded + value.length;

    if (curLoaded > total) {
      total = curLoaded;

      const extendedBufer = new Uint8Array(total);

      extendedBufer.set(buffer);

      buffer = extendedBufer;
    }

    buffer.set(value, loaded);
    loaded = curLoaded;

    progressCallback(parseInt(loaded*100/total));

    return read();
  };
}