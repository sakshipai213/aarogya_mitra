// App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Language from "./pages/Language";
import Website from "./pages/Website";
import PatientCamera from "./pages/PatientCamera";
import Pharmacist from "./pages/Pharmacist";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Language />} />
        <Route path="/home" element={<Website />} />
        <Route path="/camera" element={<PatientCamera />} />
        <Route path="/pharmacist" element={<Pharmacist />} />
      </Routes>
    </Router>
  );
}


export default App;
