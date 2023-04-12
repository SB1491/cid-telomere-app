import { StyleSheet, Text, View, Pressable, Image, Button } from 'react-native'
import { MobileModel } from 'react-native-pytorch-core'
import * as ImagePicker from "expo-image-picker"
import { useState } from 'react'
import styles from '../../assets/styles'
import { Header } from 'react-native-elements'

const Main = () => {
  const [status, requestPermission] = ImagePicker.useMediaLibraryPermissions()
  const [imageUrl, setImageUrl] = useState('')

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
      quality: 1,
      
    })

    if (result.canceled) {
      return null
    }

    setImageUrl(result.assets[0].uri)
  }

  return (
    <>
      <Header
        placement="left"
        leftComponent={{ icon: 'menu', color: '#fff' }}
        centerComponent={{ text: 'TELOMERE APP', style: { color: '#fff' } }}
        rightComponent={{ icon: 'home', color: '#fff' }}
      />
      <View style={styles.container}>
        {imageUrl && <Image source={{uri: imageUrl}} style={{ width: 200, height: 200 }}/>}
        <Button
          onPress={loadImage}
          title="Upload image"
        />
      </View>
    </>
  )
}

export default Main
