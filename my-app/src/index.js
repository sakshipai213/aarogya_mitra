// index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { TranslatorProvider } from "./components/Translator";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <TranslatorProvider>
    <App />
  </TranslatorProvider>
);
