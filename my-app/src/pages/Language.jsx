// Language.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import useTranslator from "../components/Translator";

function Language() {
  const navigate = useNavigate();
  const { setLanguage } = useTranslator();

  const handleLanguageChange = (lang) => {
    setLanguage(lang);
    navigate("/home");
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Select Language</h2>
      <button onClick={() => handleLanguageChange("en")}>English</button>
      <button onClick={() => handleLanguageChange("hi")}>Hindi</button>
    </div>
  );
}

export default Language;
