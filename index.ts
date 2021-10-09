import * as faceapi from 'face-api.js';
import '@tensorflow/tfjs-node'; // import this for speedup

// SsdMobilenetv1Options
const minConfidence = 0.5

// TinyFaceDetectorOptions
const scoreThreshold = 0.5


const ssdOptions = new faceapi.SsdMobilenetv1Options({ minConfidence })
const tinyFaceOptions = new faceapi.TinyFaceDetectorOptions({  scoreThreshold })

const canvas = require('canvas')
// import  { Canvas, Image, ImageData }  from 'canvas';

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

async function run() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('weights')
  await faceapi.nets.tinyFaceDetector.loadFromDisk('weights')
  await faceapi.nets.faceLandmark68Net.loadFromDisk('weights')
  await faceapi.nets.ageGenderNet.loadFromDisk('weights')
  
  const img = await canvas.loadImage(process.argv[2])
  var results = await faceapi.detectAllFaces(img, ssdOptions)
    .withFaceLandmarks()
    .withAgeAndGender()

  if(results.length==0){
    results = await faceapi.detectAllFaces(img,tinyFaceOptions)
              .withFaceLandmarks()
              .withAgeAndGender()
  }
  console.log(results)
}

run()
