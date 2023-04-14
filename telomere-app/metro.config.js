// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');
const defaultAssetExts = require('metro-config/src/defaults/defaults').assetExts;

module.exports = {
  ...getDefaultConfig(__dirname),
  resolver: {
    assetExts: [...defaultAssetExts, 'bin'],
  }
}