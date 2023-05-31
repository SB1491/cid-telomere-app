import { Dispatch, createContext, useContext, useReducer } from 'react'

type Action = { type: 'set', metadata: number[] | null }

type MetadataDispatch = Dispatch<Action>;

const MetadataContext = createContext<number[] | null>(null)
const MetadataDispatchContext = createContext<MetadataDispatch | null>(null)

export function MetadataProvider({ children }) {
  const [metadata, dispatch] = useReducer(
    metadataReducer,
    initialMetadata
  )

  return (
    <MetadataContext.Provider value={metadata}>
      <MetadataDispatchContext.Provider value={dispatch}>
        {children}
      </MetadataDispatchContext.Provider>
    </MetadataContext.Provider>
  )
}

export function useMetadata() {
  return useContext(MetadataContext);
}

export function useMetadataDispatch() {
  return useContext(MetadataDispatchContext);
}

const metadataReducer = (metadata: number[], action: Action) => {
  switch (action.type) {
    case 'set': {
      return action.metadata
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}

const initialMetadata = null
