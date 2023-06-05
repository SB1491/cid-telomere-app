import { StyleSheet, Text, View, Image, Button, ScrollView } from 'react-native'
import { Header } from 'react-native-elements'
import { useEffect, useRef, useState } from 'react'
import * as ImagePicker from "expo-image-picker"
import * as ImageManipulator from 'expo-image-manipulator'
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native'
import { VictoryChart, VictoryArea, VictoryTheme, VictoryPolarAxis } from 'victory-native'
import styles from '../../assets/styles'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { useMetadata, useMetadataDispatch } from '../../context/MetadataContext'

const modelJson = require('../../assets/telomere_model_tfjs/model.json')
const modelWeights1 = require('../../assets/telomere_model_tfjs/group1-shard1of38.bin')
const modelWeights2 = require('../../assets/telomere_model_tfjs/group1-shard2of38.bin')
const modelWeights3 = require('../../assets/telomere_model_tfjs/group1-shard3of38.bin')
const modelWeights4 = require('../../assets/telomere_model_tfjs/group1-shard4of38.bin')
const modelWeights5 = require('../../assets/telomere_model_tfjs/group1-shard5of38.bin')
const modelWeights6 = require('../../assets/telomere_model_tfjs/group1-shard6of38.bin')
const modelWeights7 = require('../../assets/telomere_model_tfjs/group1-shard7of38.bin')
const modelWeights8 = require('../../assets/telomere_model_tfjs/group1-shard8of38.bin')
const modelWeights9 = require('../../assets/telomere_model_tfjs/group1-shard9of38.bin')
const modelWeights10 = require('../../assets/telomere_model_tfjs/group1-shard10of38.bin')
const modelWeights11 = require('../../assets/telomere_model_tfjs/group1-shard11of38.bin')
const modelWeights12 = require('../../assets/telomere_model_tfjs/group1-shard12of38.bin')
const modelWeights13 = require('../../assets/telomere_model_tfjs/group1-shard13of38.bin')
const modelWeights14 = require('../../assets/telomere_model_tfjs/group1-shard14of38.bin')
const modelWeights15 = require('../../assets/telomere_model_tfjs/group1-shard15of38.bin')
const modelWeights16 = require('../../assets/telomere_model_tfjs/group1-shard16of38.bin')
const modelWeights17 = require('../../assets/telomere_model_tfjs/group1-shard17of38.bin')
const modelWeights18 = require('../../assets/telomere_model_tfjs/group1-shard18of38.bin')
const modelWeights19 = require('../../assets/telomere_model_tfjs/group1-shard19of38.bin')
const modelWeights20 = require('../../assets/telomere_model_tfjs/group1-shard20of38.bin')
const modelWeights21 = require('../../assets/telomere_model_tfjs/group1-shard21of38.bin')
const modelWeights22 = require('../../assets/telomere_model_tfjs/group1-shard22of38.bin')
const modelWeights23 = require('../../assets/telomere_model_tfjs/group1-shard23of38.bin')
const modelWeights24 = require('../../assets/telomere_model_tfjs/group1-shard24of38.bin')
const modelWeights25 = require('../../assets/telomere_model_tfjs/group1-shard25of38.bin')
const modelWeights26 = require('../../assets/telomere_model_tfjs/group1-shard26of38.bin')
const modelWeights27 = require('../../assets/telomere_model_tfjs/group1-shard27of38.bin')
const modelWeights28 = require('../../assets/telomere_model_tfjs/group1-shard28of38.bin')
const modelWeights29 = require('../../assets/telomere_model_tfjs/group1-shard29of38.bin')
const modelWeights30 = require('../../assets/telomere_model_tfjs/group1-shard30of38.bin')
const modelWeights31 = require('../../assets/telomere_model_tfjs/group1-shard31of38.bin')
const modelWeights32 = require('../../assets/telomere_model_tfjs/group1-shard32of38.bin')
const modelWeights33 = require('../../assets/telomere_model_tfjs/group1-shard33of38.bin')
const modelWeights34 = require('../../assets/telomere_model_tfjs/group1-shard34of38.bin')
const modelWeights35 = require('../../assets/telomere_model_tfjs/group1-shard35of38.bin')
const modelWeights36 = require('../../assets/telomere_model_tfjs/group1-shard36of38.bin')
const modelWeights37 = require('../../assets/telomere_model_tfjs/group1-shard37of38.bin')
const modelWeights38 = require('../../assets/telomere_model_tfjs/group1-shard38of38.bin')
const ioHandler = bundleResourceIO(
  modelJson,
  [
    modelWeights1,
    modelWeights2,
    modelWeights3,
    modelWeights4,
    modelWeights5,
    modelWeights6,
    modelWeights7,
    modelWeights8,
    modelWeights9,
    modelWeights10,
    modelWeights11,
    modelWeights12,
    modelWeights13,
    modelWeights14,
    modelWeights15,
    modelWeights16,
    modelWeights17,
    modelWeights18,
    modelWeights19,
    modelWeights20,
    modelWeights21,
    modelWeights22,
    modelWeights23,
    modelWeights24,
    modelWeights25,
    modelWeights26,
    modelWeights27,
    modelWeights28,
    modelWeights29,
    modelWeights30,
    modelWeights31,
    modelWeights32,
    modelWeights33,
    modelWeights34,
    modelWeights35,
    modelWeights36,
    modelWeights37,
    modelWeights38,
  ]
)


const MainPage = ({ navigation }) => {
  const [status, requestPermission] = ImagePicker.useMediaLibraryPermissions()
  const [image, setImage] = useState<ImagePicker.ImagePickerAsset>(null)
  const [ready, setReady] = useState<Boolean>(false)
  const [waiting, setWaiting] = useState<Boolean>(false)
  const [running, setRunning] = useState<Boolean>(false)
  const [result, setResult] = useState<Number[]>([0, 0, 0, 0, 0, 0])
  const model = useRef<tf.GraphModel>(null)
  const metadata = useMetadata()
  const dispatch = useMetadataDispatch()

  /* load metadata at starting time */
  useEffect(() => {
    (async () => {
      try {
        const jsonData = await AsyncStorage.getItem("@metadata")
        if (jsonData === null) {
          navigation.navigate('Setting')
        } else {
          dispatch({
            type: "set",
            metadata: JSON.parse(jsonData) as number[]
          })
        }
      } catch (error) {
        console.error(error)
      }
    })()
  }, [])

  /* load AI model at starting time */
  useEffect(() => {
    (async () => {
      await tf.ready()
      model.current = await loadGraphModel(ioHandler)
      setReady(true)
    })()
  }, [])
  
  /* 
   * if an image is uploaded before model loading,
   * start inferenece after model loading
   */
  useEffect(() => {
    if (ready && waiting) {
      setWaiting(false)
      inference(image)
    }
  }, [ready, waiting])

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
    const mean = tf.tensor([[[0.485, 0.456, 0.406]]])
    const std = tf.tensor([[[0.229, 0.224, 0.225]]])
    const resizedTensor = tf.image.resizeBilinear(imageTensor, [232, 232])
    const normalizedTensor = resizedTensor.div(255).sub(mean).div(std)
    const expandedTensor = tf.expandDims(normalizedTensor, 0)

    return expandedTensor
  }

  const inference = async (image: ImagePicker.ImagePickerAsset) => {
    try {
      setRunning(true)

      const imgTensor = await imageToTensor(image)

      const inputMetadata = tf.tensor([metadata])

      const score = model.current.predict([imgTensor, inputMetadata]) as tf.Tensor<tf.Rank>

      const maxIdx = tf.argMax(tf.reshape(score, [6, 4]), 1).arraySync() as number[]

      setResult(maxIdx)
    }
    finally {
      setRunning(false)
    }
  }

  const loadImage = async () => {
    if (metadata === null) {
      navigation.navigate('Setting')
      return
    }
    
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
    <>
      <Header
        placement="left"
        leftComponent={{ icon: 'menu', color: '#fff', onPress: () => navigation.openDrawer() }}
        centerComponent={{ text: 'TELOMERE APP', style: { color: '#fff' } }}
        rightComponent={{ icon: 'home', color: '#fff' }}
      />
      <View style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <Text></Text>
          <Text>{ready ? "Model is ready!\n" : "Loading model...\n"}</Text>

          {image && <Image source={{uri: image.uri}} style={{ width: 232, height: 232 }}/>}
          <Text></Text>
          
          <Button
            onPress={loadImage}
            title="Upload image"
          />
          <Text>{"\n"}</Text>

          {waiting
            ? <Text>Waiting for model to load...</Text> 
            : null
          }
          {running
            ? <Text>Running...</Text>
            : result 
              ? <VictoryChart
                  polar
                  theme={VictoryTheme.material}
                  style={{
                    background: { fill: "#fff" }
                  }}
                >
                  <VictoryArea 
                    data={[
                    {'x': '탈모', 'y': result[5]},
                    {'x': '미세각질', 'y': result[0]},
                    {'x': '피지과다', 'y': result[1]},
                    {'x': '모낭사이홍반', 'y': result[2]},
                    {'x': '모낭홍반/농포', 'y': result[3]},
                    {'x': '비듬', 'y': result[4]},
                    ]}
                    style={{
                      data: {fill: "skyblue"}
                    }}
                  />
                  <VictoryPolarAxis
                    labelPlacement='vertical'
                  />
                  
                  <VictoryPolarAxis dependentAxis
                    tickValues={[0, 1, 2, 3]}
                    axisAngle={90}
                    labelPlacement='vertical'
                  />
                </VictoryChart>
              : null
          }
        </ScrollView>
      </View>
    </>
  )
}

export default MainPage
