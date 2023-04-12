import { StyleSheet, Text, View, Pressable, Image } from 'react-native'
import { MobileModel } from 'react-native-pytorch-core'
import * as ImagePicker from "expo-image-picker"
import { useState } from 'react'

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
    <View style={styles.container}>
      {imageUrl && <Image source={{uri: imageUrl}} style={{ width: 200, height: 200 }}/>}
      <Pressable onPress={loadImage}>
        <Text>
          Upload image
        </Text>
      </Pressable>
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
