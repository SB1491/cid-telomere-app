import { NavigationContainer } from "@react-navigation/native"
import { createDrawerNavigator } from "@react-navigation/drawer"

import MainPage from "./container/MainPage/MainPage"
import SettingPage from "./container/SettingPage/Settingpage"
import { MetadataProvider } from "./context/MetadataContext"

const Drawer = createDrawerNavigator();

const App = () => {
  return (
    <MetadataProvider>
      <NavigationContainer>
        <Drawer.Navigator initialRouteName="Main">
          <Drawer.Screen name="Main" component={MainPage} />
          <Drawer.Screen name="Setting" component={SettingPage} />
        </Drawer.Navigator>
      </NavigationContainer>
    </MetadataProvider>
  )
}

export default App
