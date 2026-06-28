import React, { useRef, useState, useEffect } from "react";



function Pharmacist() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [language, setLanguage] = useState("");
  const [medicines, setMedicines] = useState(
    Array.from({ length: 6 }, () => ({
      medicine: "",
      route: "",
      dosageDuration: "",
      substitutes: [],    // array of substitute strings returned by backend
      subLoading: false,
      subError: ""
    }))
  );

  // Start webcam on load
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((err) => console.error("Camera error:", err));
  }, []);

  // Capture from webcam
  const capturePhoto = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL("image/png");
    setCapturedImage(dataURL);
    setUploadedImage(null);
  };

  // Upload from device and auto-fill from backend (existing /generate endpoint)
  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result);
      setCapturedImage(null);
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/generate", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      const extracted = (data || []).map((item) => ({
        medicine: item.medicine || "",
        route: item.route || "",
        dosageDuration: item.dosage || "",
        substitutes: [],
        subLoading: false,
        subError: ""
      }));

      // Ensure we always keep at least one row (or adjust as you prefer)
      setMedicines(extracted.length ? extracted : [{ medicine: "", route: "", dosageDuration: "", substitutes: [], subLoading: false, subError: "" }]);
    } catch (err) {
      console.error("Backend error:", err);
      alert("Failed to fetch medicine info.");
    }
  };

  // Handle input changes
  const handleMedicineChange = (index, e) => {
    const updated = [...medicines];
    updated[index][e.target.name] = e.target.value;
    // Clear previous substitutes if user edits medicine (optional)
    updated[index].substitutes = [];
    updated[index].subError = "";
    setMedicines(updated);
  };

  // Helper to parse backend substitute label (if it contains "➜ matched in A as '...'", prefer canonical match)
  const parseCanonicalName = (label) => {
    try {
      // look for pattern like:  "subname  ➜  matched in A as 'matched_name'"
      const match = label.match(/matched in A as\s+'(.+)'/i);
      if (match && match[1]) return match[1];
      // fallback: if label has "➜", use left side (substitute raw)
      if (label.includes("➜")) return label.split("➜")[0].trim();
      return label;
    } catch {
      return label;
    }
  };

  // Find substitutes for a given medicine row
  const findSubstitute = async (index) => {
    const medName = medicines[index].medicine.trim();
    if (!medName) {
      alert("Enter a medicine name first.");
      return;
    }

    // set loading
    setMedicines((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], subLoading: true, subError: "", substitutes: [] };
      return next;
    });

    try {
      const res = await fetch("http://127.0.0.1:5000/find_substitute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ medicine_name: medName })
      });

      const data = await res.json();

      if (!res.ok || data.status === "error") {
        const message = data?.message || `No substitutes found for '${medName}'.`;
        setMedicines((prev) => {
          const next = [...prev];
          next[index] = { ...next[index], subLoading: false, subError: message, substitutes: [] };
          return next;
        });
        return;
      }

      // backend (Flask example) returns { status: "success", matched_med, valid_substitutes }
      const rawSubs = data.valid_substitutes || data.substitutes || [];
      setMedicines((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], subLoading: false, subError: "", substitutes: rawSubs };
        return next;
      });
    } catch (err) {
      console.error("Substitute fetch error:", err);
      setMedicines((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], subLoading: false, subError: "Failed to fetch substitutes.", substitutes: [] };
        return next;
      });
    }
  };

  // When user clicks a substitute chip, fill the medicine input with the canonical name
  const handleSubstituteClick = (index, label) => {
    const canonical = parseCanonicalName(label);
    setMedicines((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], medicine: canonical };
      return next;
    });
  };

  const addMedicine = () => {
    setMedicines([...medicines, { medicine: "", route: "", dosageDuration: "", substitutes: [], subLoading: false, subError: "" }]);
  };

  const removeMedicine = (index) => {
    const updated = [...medicines];
    updated.splice(index, 1);
    setMedicines(updated.length ? updated : [{ medicine: "", route: "", dosageDuration: "", substitutes: [], subLoading: false, subError: "" }]);
  };

  // Send to backend and download instructions (unchanged)
  const handleGenerate = async () => {
    if (!language) {
      alert("Please select a language");
      return;
    }

    const structuredInput = medicines.map((m) => (
      `${m.medicine};${m.route};${m.dosageDuration}`
    )).join("\n");

    try {
      const response = await fetch("http://127.0.0.1:5000/generate-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          structured_input: structuredInput,
          language: language
        })
      });

      const data = await response.json();
      const content = `English:\n${data.english_instruction}\n\nTranslated:\n${data.translated_instruction}`;
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'TranslatedInstruction.txt';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("API Error:", error);
      alert("Something went wrong. Check console.");
    }
  };

  // Simple inline styles used to keep file self-contained
  const styles = {
    container: { padding: 20, maxWidth: 760, margin: "auto" },
    row: { marginBottom: 20, padding: 12, border: "1px solid #ccc", borderRadius: 8, background: "#fff" },
    topRow: { display: "flex", gap: 12, alignItems: "center" },
    leftGrow: { flex: 1, display: "flex", gap: 8, alignItems: "center" },
    substitutesContainer: { minWidth: 220, display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" },
    chip: { padding: "6px 8px", borderRadius: 16, background: "#f0f0f0", cursor: "pointer", fontSize: 13, border: "1px solid #e0e0e0" },
    findBtn: { background: "#FFA500", color: "white", border: "none", padding: "6px 10px", borderRadius: 4, cursor: "pointer" },
    removeBtn: { marginTop: 8, background: "#f44336", color: "white", border: "none", padding: "5px 10px", borderRadius: 4, cursor: "pointer" }
  };

  return (
    <div style={styles.container}>
      <h2>📸 Capture or Upload Prescription</h2>

      <video ref={videoRef} autoPlay playsInline style={{ width: "100%", borderRadius: 8 }} />
      <div style={{ marginTop: 10 }}>
        <button onClick={capturePhoto} style={{ marginRight: 8 }}>📷 Take Photo</button>
        <input type="file" accept="image/*" onChange={handleUpload} />
      </div>

      {(capturedImage || uploadedImage) && (
        <div style={{ marginTop: 15 }}>
          <img
            src={capturedImage || uploadedImage}
            alt="Prescription"
            style={{ width: "100%", borderRadius: 8 }}
          />
        </div>
      )}

      <h3 style={{ marginTop: 20 }}>🌐 Choose Language</h3>
      <select
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
        style={{ width: "100%", padding: 8, marginBottom: 10 }}
      >
        <option value="">Select a language</option>
        <option value="english">English</option>
        <option value="hindi">Hindi</option>
        <option value="kannada">Kannada</option>
        <option value="tamil">Tamil</option>
        <option value="telugu">Telugu</option>
        <option value="malayalam">Malayalam</option>
        <option value="marathi">Marathi</option>
        <option value="gujarati">Gujarati</option>
        <option value="bengali">Bengali</option>
        <option value="urdu">Urdu</option>
      </select>

      <div style={{ marginBottom: 20 }}>
        <button
          onClick={handleGenerate}
          style={{
            background: "#008CBA",
            color: "white",
            padding: "10px 16px",
            border: "none",
            borderRadius: 4,
            cursor: "pointer"
          }}
        >
          🚀 Generate
        </button>
      </div>

      <h3 style={{ marginTop: 10 }}>📝 Prescription Info</h3>

      {medicines.map((med, index) => (
        <div key={index} style={styles.row}>
          <strong>Medicine {index + 1}</strong>

          <div style={{ ...styles.topRow, marginTop: 8 }}>
            <div style={styles.leftGrow}>
              <input
                type="text"
                name="medicine"
                placeholder="Medicine Name"
                value={med.medicine}
                onChange={(e) => handleMedicineChange(index, e)}
                style={{ flex: 1, padding: 8 }}
              />
              <button
                onClick={() => findSubstitute(index)}
                style={styles.findBtn}
                disabled={med.subLoading}
                title="Find substitute medicines"
              >
                {med.subLoading ? "Searching..." : "🔍 Find Substitute"}
              </button>
            </div>

            {/* substitutes shown inline to the right */}
            <div style={styles.substitutesContainer}>
              {med.subError ? (
                <div style={{ color: "#b00", fontSize: 13 }}>{med.subError}</div>
              ) : (
                med.substitutes.map((s, i) => (
                  <div
                    key={i}
                    onClick={() => handleSubstituteClick(index, s)}
                    style={styles.chip}
                    title={`Click to use "${parseCanonicalName(s)}"`}
                  >
                    {s}
                  </div>
                ))
              )}
            </div>
          </div>

          <input
            type="text"
            name="route"
            placeholder="Route (e.g., oral, eye)"
            value={med.route}
            onChange={(e) => handleMedicineChange(index, e)}
            style={{ width: "100%", marginTop: 12, padding: 8 }}
          />
          <input
            type="text"
            name="dosageDuration"
            placeholder="Dosage & Duration (e.g., 500mg for 5 days)"
            value={med.dosageDuration}
            onChange={(e) => handleMedicineChange(index, e)}
            style={{ width: "100%", marginTop: 8, padding: 8 }}
          />

          <button
            onClick={() => removeMedicine(index)}
            style={styles.removeBtn}
          >
            🗑 Remove
          </button>
        </div>
      ))}

      <div style={{ marginTop: 12 }}>
        <button onClick={addMedicine} style={{ background: "#4CAF50", color: "white", border: "none", padding: "8px 12px", borderRadius: 4 }}>
          ➕ Add Medicine
        </button>
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

export default Pharmacist;
