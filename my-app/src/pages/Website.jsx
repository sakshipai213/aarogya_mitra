// Website.jsx
import React from "react";
import useTranslator from "../components/Translator";
import { useNavigate } from "react-router-dom";

function Website() {
  const { t } = useTranslator();
  const navigate = useNavigate();

  return (
    <div style={{ padding: 20 }}>
      <h1 style={{ textAlign: "center" }}>{t("welcome")}</h1>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 400, margin: "auto" }}>
        <button onClick={() => navigate("/camera")}>{t("patient_button")}</button>
        <button onClick={() => navigate("/pharmacist")}>{t("pharmacist_button")}</button>
      </div>
    </div>
  );
}

export default Website;
