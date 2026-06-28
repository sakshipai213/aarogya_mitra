// components/Translator.js
import React, { createContext, useContext, useState } from "react";
import translations from "../translations";



const TranslatorContext = createContext();

export function TranslatorProvider({ children }) {
  const [language, setLanguage] = useState("en");

  const t = (key) => {
    return translations[language]?.[key] || key;
  };

  return (
    <TranslatorContext.Provider value={{ t, setLanguage }}>
      {children}
    </TranslatorContext.Provider>
  );
}

export default function useTranslator() {
  return useContext(TranslatorContext);
}
