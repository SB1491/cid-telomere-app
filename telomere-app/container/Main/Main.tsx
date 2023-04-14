import { StyleSheet, Text, View, Pressable, Image } from 'react-native'
import { useEffect, useRef, useState } from 'react'
import * as ImagePicker from "expo-image-picker"
import * as ImageManipulator from 'expo-image-manipulator'
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native'

const modelJson = require('../../assets/telomere_model_tfjs/model.json')
const modelWeights1 = require('../../assets/telomere_model_tfjs/group1-shard1of4.bin')
const modelWeights2 = require('../../assets/telomere_model_tfjs/group1-shard2of4.bin')
const modelWeights3 = require('../../assets/telomere_model_tfjs/group1-shard3of4.bin')
const modelWeights4 = require('../../assets/telomere_model_tfjs/group1-shard4of4.bin')

const ioHandler = bundleResourceIO(
  modelJson,
  [modelWeights1, modelWeights2, modelWeights3, modelWeights4]
)


const idx_to_result = [
  "Normal(0)",
  "Middle(1)",
  "Moderate(2)",
  "Severe(3)"
]

const mean = tf.tensor([[[0.485, 0.456, 0.406]]])
const std = tf.tensor([[[0.229, 0.224, 0.225]]])


const Main = () => {
  const [status, requestPermission] = ImagePicker.useMediaLibraryPermissions()
  const [image, setImage] = useState<ImagePicker.ImagePickerAsset>(null)
  const [ready, setReady] = useState<Boolean>(false)
  const [waiting, setWaiting] = useState<Boolean>(false)
  const [running, setRunning] = useState<Boolean>(false)
  const [result, setResult] = useState<String>('')
  const model = useRef<tf.GraphModel>(null)

  useEffect(() => {
    (async () => {
      await tf.ready()
      model.current = await loadGraphModel(ioHandler)
      setReady(true)
    })()
  }, [])
  
  useEffect(() => {
    if (ready && waiting) {
      setWaiting(false)
      inference(image)
    }
  }, [ready])

  const imageToTensor = async (image: ImagePicker.ImagePickerAsset) => {
    // convert image to 232*232, JPEG format
    const convertedImage = await ImageManipulator.manipulateAsync(
      image.uri,
      [{ resize: { height: 232, width: 232 } }],
      { base64: true, compress: 1, format: ImageManipulator.SaveFormat.JPEG }
    )

    // image base64 to tensor
    const imgBuffer = tf.util.encodeString(convertedImage.base64, 'base64').buffer
    const raw = new Uint8Array(imgBuffer)
    const imageTensor = decodeJpeg(raw)

    // resize to [1, 232, 232, 3]
    const resizedTensor = tf.image.resizeBilinear(imageTensor, [232, 232])
    const normalizedTensor = resizedTensor.div(255).sub(mean).div(std)
    const expandedTensor = tf.expandDims(normalizedTensor, 0)

    return expandedTensor
  }

  const inference = async (image: ImagePicker.ImagePickerAsset) => {
    try {
      setRunning(true)

      const imgTensor = await imageToTensor(image)

      const score = model.current.predict(imgTensor) as tf.Tensor<tf.Rank>

      const maxIdx = tf.argMax(tf.squeeze(score), 0).arraySync() as number

      setResult(idx_to_result[maxIdx])
    }
    finally {
      setRunning(false)
    }
  }

  const loadImage = async () => {
    if (!status?.granted) {
      const permission = await requestPermission()
      if (!permission.granted) {
        return null
      }
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1
    })

    if (result.canceled) {
      return null
    }

    const resultImage = result.assets[0]

    setImage(resultImage)

    if (ready) {
      inference(resultImage)
    } else {
      setWaiting(true)
    }
  }

  return (
    <View style={styles.container}>
      <Text>{ready ? "Model is ready!" : "Loading model..."}</Text>
      <Text>{"\n"}</Text>

      {image && <Image source={{uri: image.uri}} style={{ width: 232, height: 232 }}/>}
      <Text>{"\n"}</Text>

      <Pressable onPress={loadImage}>
        <Text>
          Upload image
        </Text>
      </Pressable>
      <Text>{"\n"}</Text>
      
      {waiting
        ? <Text>Waiting for model to load...</Text> 
        : null
      }
      {running
        ? <Text>Running...</Text>
        : result 
          ? <Text>Prediction result: {result}</Text> 
          : null
      }
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default Main
