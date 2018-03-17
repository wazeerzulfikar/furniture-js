
import * as dl from 'deeplearn';

import {NDArray} from 'deeplearn';
import {TensorflowLoader} from 'deeplearn-tensorflow';

// const tensorflowReader = new TensorflowLoader(NDArray);
// tensorflowReader.loadRemoteFiles('tf_model/model.ckpt-200').then((vars) => {
//     console.log('Done');
// }); 

// manifest.json lives in the same directory as the mnist demo.
const label_strings = ['bed', 'chair', 'lamp', 'shelf', 'sofa', 'stool', 'table', 'wardrobe']

const reader = new dl.CheckpointLoader('model_full');
reader.getAllVariables().then(vars => {
  // Get sample data.
  const xhr = new XMLHttpRequest();
  xhr.open('GET', 'test_furniture_var_size_data.json');
  xhr.onload = async () => {
    const data = JSON.parse(xhr.responseText) as SampleData;

    console.log(`Evaluation set: n=${data.images.length}.`);

    let numCorrect = 0;
    for (let i = 0; i < data.images.length; i++) {

      const x = dl.tensor3d(data.images[i]);
      const img_size = Math.sqrt(x.shape[0]/3);

      const inferred = dl.tidy(() => {
        console.log(`Item ${i}, with size of ${img_size} x ${img_size}.`);

        return infer(x, vars);
        // Infer through the model to get a prediction.
      });

      const predictedLabel = Math.round(await inferred.val());
      inferred.dispose();
      console.log(`Item ${i}, predicted label ${predictedLabel}.`);

      const label = data.labels[i];
      if (label === predictedLabel) {
        numCorrect++;
      }

      // Show the image.
      dl.tidy(() => {
        const result =
            renderResults(dl.tensor1d(data.images[i]), label_strings[label], label_strings[predictedLabel], img_size);
        document.body.appendChild(result);
      });
    }

    // Compute final accuracy.
    const accuracy = numCorrect * 100 / data.images.length;
    document.getElementById('accuracy').innerHTML = `${accuracy}%`;
  };
  xhr.onerror = (err) => console.error(err);
  xhr.send();

  console.log("Loaded")
});

export interface SampleData {
  images: number[][];
  labels: number[];
}


export function infer(
    x: dl.Tensor3D, vars: {[varName: string]: dl.Tensor}): dl.Scalar {

  const conv1B = vars['ConvNet/conv2d/bias'] as dl.Tensor1D;
  const conv1W = vars['ConvNet/conv2d/kernel'] as dl.Tensor4D;
  const conv2B = vars['ConvNet/conv2d_1/bias'] as dl.Tensor1D;
  const conv2W = vars['ConvNet/conv2d_1/kernel'] as dl.Tensor4D;
  const conv3B = vars['ConvNet/conv2d_2/bias'] as dl.Tensor1D;
  const conv3W = vars['ConvNet/conv2d_2/kernel'] as dl.Tensor4D;
  const conv4B = vars['ConvNet/conv2d_3/bias'] as dl.Tensor1D;
  const conv4W = vars['ConvNet/conv2d_3/kernel'] as dl.Tensor4D;
  const conv5B = vars['ConvNet/conv2d_4/bias'] as dl.Tensor1D;
  const conv5W = vars['ConvNet/conv2d_4/kernel'] as dl.Tensor4D;
  const conv6B = vars['ConvNet/conv2d_5/bias'] as dl.Tensor1D;
  const conv6W = vars['ConvNet/conv2d_5/kernel'] as dl.Tensor4D;
  const conv7B = vars['ConvNet/conv2d_6/bias'] as dl.Tensor1D;
  const conv7W = vars['ConvNet/conv2d_6/kernel'] as dl.Tensor4D;

  const hidden1B = vars['ConvNet/dense/bias'] as dl.Tensor1D;
  const hidden1W = vars['ConvNet/dense/kernel'] as dl.Tensor2D;
  const hidden2B = vars['ConvNet/dense_1/bias'] as dl.Tensor1D;
  const hidden2W = vars['ConvNet/dense_1/kernel'] as dl.Tensor2D;
  // const softmaxW = vars['softmax_linear/weights'] as dl.Tensor2D;
  // const softmaxB = vars['softmax_linear/biases'] as dl.Tensor1D;

  var shape = x.shape
  var img_size = Math.sqrt(shape[0]/3)

  const inp = x.as4D(-1,img_size,img_size,3);

  const conv1 = dl.conv2d(inp, conv1W, 1,'valid')
                  .add(conv1B)
                  .relu() as dl.Tensor4D;

  const conv2 = dl.conv2d(conv1, conv2W, 2,'valid')
                  .add(conv2B)
                  .relu() as dl.Tensor4D;

  const conv3 = dl.conv2d(conv2, conv3W, 1,'valid')
                  .add(conv3B)
                  .relu() as dl.Tensor4D;
   
  const conv4 = dl.conv2d(conv3, conv4W, 2,'valid')
                  .add(conv4B)
                  .relu() as dl.Tensor4D;
   
  const conv5 = dl.conv2d(conv4, conv5W, 1,'valid')
                  .add(conv5B)
                  .relu() as dl.Tensor4D;
   
  const conv6 = dl.conv2d(conv5, conv6W, 2,'valid')
                  .add(conv6B)
                  .relu() as dl.Tensor4D;
   
  const conv7 = dl.conv2d(conv6, conv7W, 1,'valid')
                  .add(conv7B)
                  .relu() as dl.Tensor4D;

  const globalmaxpool = dl.max(conv7, [1,2]) as dl.Tensor2D;

  const hidden1 = globalmaxpool.matMul(hidden1W).add(hidden1B) as dl.Tensor1D;

  const hidden2 = hidden1.as2D(-1, hidden2W.shape[0]).matMul(hidden2W).add(hidden2B) as dl.Tensor1D;

  const logits = hidden2.softmax() as dl.Tensor1D;

  console.log(logits.shape)

  return logits.argMax();
}

function renderImage(array: dl.Tensor1D, image_size: number) {
  const width = image_size;
  const height = image_size;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const float32Array = array.dataSync();
  const imageData = ctx.createImageData(width, height);
  for (let i = 0, j = 0; i < float32Array.length; i+=3, j+=4) {
    const r_value = Math.round(float32Array[i] * 255);
    const g_value = Math.round(float32Array[i+1] * 255);
    const b_value = Math.round(float32Array[i+2] * 255);

    imageData.data[j + 0] = r_value;
    imageData.data[j + 1] = g_value;
    imageData.data[j + 2] = b_value;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function renderResults(
    array: dl.Tensor1D, label: string, predictedLabel: string, image_size: number) {
  const root = document.createElement('div');
  root.appendChild(renderImage(array, image_size));
  const img_size = document.createElement('div');
  img_size.innerHTML = `Original Image Size: ${image_size} x ${image_size}`;
  root.appendChild(img_size);
  const actual = document.createElement('div');
  actual.innerHTML = `Actual: ${label}`;
  root.appendChild(actual);
  const predicted = document.createElement('div');
  predicted.innerHTML = `Predicted: ${predictedLabel}`;
  root.appendChild(predicted);

  if (label !== predictedLabel) {
    root.classList.add('error');
  }

  root.classList.add('result');
  return root;
}
