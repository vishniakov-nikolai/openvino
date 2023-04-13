var ov = require('./lib/ov_node_addon.node');;


const fs = require('fs');
const math = require('./lib/math_func.js');
const Jimp = require('jimp');


async function create_tensor(img_path) {
    const jimpSrc = await Jimp.read(img_path);
    const src = cv.matFromImageData(jimpSrc.bitmap);   
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
    cv.resize(src, src, new cv.Size(224, 224));
    //create tensor
    const tensor_data = new Float32Array(src.data);
    const tensor = new ov.Tensor(ov.element.f32, Int32Array.from([1, 224, 224, 3]), tensor_data);    

    return tensor;    
}


async function onRuntimeInitialized()
{
    // FIXME: replace mjs usage to json
    const { default: imagenetClassesMap } = await import('../assets/imagenet_classes_map.mjs');
    const img_path = process.argv[2] || '../assets/images/shih_tzu.jpg';
    const model_path = '../assets/models/v3-small_224_1.0_float.xml';
    const core = new ov.Core();

    /*   ---Read model asynchronously and create a promise---   */
    
    const model_promise = core.read_model_async(model_path);

    /*   ---Create a promise with tensor---   */
    const tensor_promise = create_tensor(img_path)

    Promise.all([model_promise, tensor_promise]).then(([model, tensor]) => {
        const output = model.compile("CPU").infer(tensor);
        //show the results
        console.log("Result: " + imagenetClassesMap[math.argMax(output.data)]);
    });

}



Module = {
    onRuntimeInitialized
};
cv = require('./lib/opencv.js');
