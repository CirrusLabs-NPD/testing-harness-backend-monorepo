import React, { createContext, useReducer } from "react";

const initialState = {
  models: [
    { id: 1, name: "Google T5", description: "Model 1" },
    { id: 2, name: "Facebook NLLB", description: "Model 2" },
    { id: 3, name: "Helsinki Opus", description: "Model 3" },
  ],
};

export const AppContext = createContext();

export const AppProvider = (props) => {
  const [state] = useReducer(initialState);

  return (
    <AppContext.Provider
      value={{
        model1: state.model1,
        model2: state.model2,
        model3: state.model3,
      }}
    >
      {props.children}
    </AppContext.Provider>
  );
};
