import { Text, View, Button } from 'react-native'
import { useState } from 'react'
import { Slider } from '@miblanchard/react-native-slider'
import { useMetadataDispatch } from '../../context/MetadataContext'
import AsyncStorage from '@react-native-async-storage/async-storage'
import styles from '../../assets/styles'


const SettingPage = ({ navigation }) => {
  const [stage, setStage] = useState<0 | 1 | 2 | 3 | 4>(0)

  const [gender, setGender] = useState<0 | 1>(0)
  const [age, setAge] = useState<number>(0)
  const [shampoo, setShampoo] = useState<1 | 2 | 4>(1)
  const [dyed, setDyed] = useState<0 | 1>(0)

  const dispatch = useMetadataDispatch()

  const goBackHandler = async () => {
    const metadata = [
      gender,
      Math.min(age / 70, 1),
      shampoo / 4,
      dyed
    ]

    try {
      await AsyncStorage.setItem("@metadata", JSON.stringify(metadata))
      dispatch({
        type: "set",
        metadata: metadata
      })
    } catch (error) {
      console.error(error)
    }

    setStage(0)
    navigation.navigate('Main')
  }

  return (
    <View style={styles.container}>
      {(() => {
        switch (stage) {
          case 0: {
            return (
              <>
                <Text>What's your gender?{"\n"}</Text>
                <View style={styles.buttonContainer}>
                  <View style={styles.button}>
                    <Button
                      onPress={() => {setGender(1); setStage(1)}}
                      title="Men"
                    />
                  </View>
                  <View style={styles.button}>
                    <Button
                      onPress={() => {setGender(0); setStage(1)}}
                      title="Women"
                    />
                  </View>
                </View>
              </>
            )
          }
          case 1: {
            return (
              <>
                <Text>What's your age?</Text>
                <Text>{age}</Text>

                <View style={styles.slide}>
                  <Slider
                    value={age}
                    animateTransitions={true}
                    thumbTouchSize={{ width: 50, height: 50 }}
                    onValueChange={value => setAge(value[0])}
                    minimumValue={0}
                    maximumValue={100}
                    step={1}
                  />
                </View>

                <Text></Text>
                <Button 
                  onPress={() => {setStage(2)}}
                  title="Confirm"
                />
              </>
            )
          }
          case 2: {
            return (
              <>
                <Text>How often do you use shampoo?{"\n"}</Text>
                <View>
                  <View style={styles.verticalButton}>
                    <Button
                      onPress={() => {setShampoo(1); setStage(3)}}
                      title="every other day"
                    />
                  </View>
                  <View style={styles.verticalButton}>
                    <Button
                      onPress={() => {setShampoo(2); setStage(3)}}
                      title="once a day"
                    />
                  </View>
                  <View style={styles.verticalButton}>
                    <Button
                      onPress={() => {setShampoo(4); setStage(3)}}
                      title="twice a day"
                    />
                  </View>
                </View>

              </>
            )
          }
          case 3: {
            return (
              <>
                <Text>Did you dye your hair?{"\n"}</Text>
                <View style={styles.buttonContainer}>
                  <View style={styles.button}>
                    <Button
                      onPress={() => {setDyed(1); setStage(4)}}
                      title="Yes"
                    />
                  </View>
                  <View style={styles.button}>
                    <Button
                      onPress={() => {setDyed(0); setStage(4)}}
                      title="No"
                    />
                  </View>
                </View>
              </>
            )
          }
          case 4: {
            return (
              <>
                <Text>Metadata settings are all over!{"\n"}</Text>
                <Button
                  onPress={() => goBackHandler()}
                  title="Save & Go back"
                />
              </>
            )
          }
        }
      })()}
    </View>
  )
}

export default SettingPage
